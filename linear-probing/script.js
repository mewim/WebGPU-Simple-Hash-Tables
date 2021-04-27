const initialSizeInput = document.getElementById("initial-size");
const resizeFactorInput = document.getElementById("resize-factor");
const resizeThresholdInput = document.getElementById("resize-threshold");
const keyRangeInput = document.getElementById("key-range");
const batchesInput = document.getElementById("batches");
const opsPerBatchInput = document.getElementById("ops-per-batch");
const runButton = document.getElementById("run");
const resultsTextArea = document.getElementById("results");

const benchmark = async (
  device,
  hashTablePipeline,
  hashTableBindGroupLayout,
  resizePipeline,
  resizeBindGroupLayout,
  RESIZE_THRESHOLD,
  INITIAL_CAPACITY,
  OPS_PER_BATCH,
  ITERATIONS,
  KEY_RANGE,
  NEW_SIZE_FACTOR
) => {
  resultsTextArea.value += `Configurations: \n`;
  resultsTextArea.value += `\tRESIZE_THRESHOLD=${RESIZE_THRESHOLD} \n`;
  resultsTextArea.value += `\tINITIAL_CAPACITY=${INITIAL_CAPACITY} \n`;
  resultsTextArea.value += `\tOPS_PER_BATCH=${OPS_PER_BATCH} \n`;
  resultsTextArea.value += `\tITERATIONS=${ITERATIONS} \n`;
  resultsTextArea.value += `\tKEY_RANGE=${KEY_RANGE} \n`;
  resultsTextArea.value += `\tNEW_SIZE_FACTOR=${NEW_SIZE_FACTOR} \n`;

  let currentCapacity = INITIAL_CAPACITY;
  let totalInsertions = 0;
  let totalDeletions = 0;
  let hashtableBufferSize = Int32Array.BYTES_PER_ELEMENT * currentCapacity;
  let hashtableBuffer = device.createBuffer({
    size: hashtableBufferSize,
    usage: GPUBufferUsage.STORAGE,
  });

  for (let i = 0; i < ITERATIONS; ++i) {
    const startTime = performance.now();
    if (totalInsertions >= currentCapacity * RESIZE_THRESHOLD) {
      const resizeStartTime = performance.now();
      let newCapacity = (totalInsertions - totalDeletions) * NEW_SIZE_FACTOR;
      if (newCapacity < OPS_PER_BATCH) {
        newCapacity = OPS_PER_BATCH;
      }
      hashtableBuffer = await runResizeCode(
        device,
        resizeBindGroupLayout,
        hashtableBuffer,
        resizePipeline,
        currentCapacity,
        newCapacity
      );
      const resizeTime = performance.now() - resizeStartTime;
      resultsTextArea.value += `Iteration ${
        i + 1
      }:\tresize completed in ${resizeTime.toFixed(
        2
      )} ms. Old capacity: ${currentCapacity},\tnew capacity: ${newCapacity}.\n`;
      currentCapacity = newCapacity;
      totalInsertions = totalInsertions - totalDeletions;
      totalDeletions = 0;
    }
    const counters = await runHashTableCode(
      device,
      hashTableBindGroupLayout,
      hashtableBuffer,
      hashTablePipeline,
      currentCapacity,
      i,
      KEY_RANGE,
      OPS_PER_BATCH
    );
    totalInsertions += counters[0];
    totalDeletions += counters[1];

    const time = performance.now() - startTime;
    resultsTextArea.value += `Iteration ${i + 1}:\t${time.toFixed(2)}ms\t${
      counters[0]
    } keys inserted,\t${counters[1]} keys deleted,\tthroughput ${(
      OPS_PER_BATCH /
      (time / 1000)
    ).toFixed(0)} ops/s.\n`;
  }
};

const runHashTableCode = async (
  device,
  bindGroupLayout,
  hashTableBuffer,
  hashTablePipeline,
  currentCapacity,
  currentIteration,
  keyRange,
  numberOfOps
) => {
  const metadataSize = 32;
  const metadataArray = new Int32Array(metadataSize);
  metadataArray[0] = currentCapacity;
  metadataArray[1] = currentIteration;
  metadataArray[2] = keyRange;
  metadataArray[3] = 0;
  metadataArray[4] = -1;

  const metadataBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: metadataSize * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });
  const mappedMetadataBuffer = metadataBuffer.getMappedRange();
  new Int32Array(mappedMetadataBuffer).set(metadataArray);
  metadataBuffer.unmap();

  const countersSize = 32;
  const countersBuffer = device.createBuffer({
    size: countersSize * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: metadataBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: countersBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: hashTableBuffer,
        },
      },
    ],
  });

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(hashTablePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(numberOfOps);
  passEncoder.endPass();

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);
  await device.queue.onSubmittedWorkDone();
  metadataBuffer.destroy();
  return await readGPUBuffer(device, countersBuffer, 8);
};

const runResizeCode = async (
  device,
  bindGroupLayout,
  oldHashTableBuffer,
  resizePipeline,
  currentCapacity,
  newCapacity
) => {
  let newHashtableBuffer = device.createBuffer({
    size: Int32Array.BYTES_PER_ELEMENT * newCapacity,
    usage: GPUBufferUsage.STORAGE,
  });
  const metadataSize = 32;
  const metadataArray = new Int32Array(metadataSize);
  metadataArray[0] = currentCapacity;
  metadataArray[1] = newCapacity;
  metadataArray[2] = 0;
  metadataArray[3] = -1;
  const metadataBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: metadataSize * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });
  const mappedMetadataBuffer = metadataBuffer.getMappedRange();
  new Int32Array(mappedMetadataBuffer).set(metadataArray);
  metadataBuffer.unmap();
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: metadataBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: oldHashTableBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: newHashtableBuffer,
        },
      },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(resizePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(currentCapacity);
  passEncoder.endPass();
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);
  await device.queue.onSubmittedWorkDone();
  oldHashTableBuffer.destroy();
  return newHashtableBuffer;
};

(async () => {
  const hashTableSource = await fetch("hash_table.glsl").then((res) =>
    res.text()
  );

  const resizeSource = await fetch("resize.glsl").then((res) => res.text());

  const glslangModule = await import(
    "https://unpkg.com/@webgpu/glslang@0.0.15/dist/web-devel/glslang.js"
  );
  const glslang = await glslangModule.default();

  if (!navigator.gpu) {
    resultsTextArea.value +=
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.\n";
      runButton.disabled = true;
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    resultsTextArea.value += "Failed to get GPU adapter.\n";
    return;
  }
  const device = await adapter.requestDevice();
  resultsTextArea.value += "WebGPU is ready.\n";

  resultsTextArea.value += "Compiling GLSL code...\n";
  const hashTableShaderCode = glslang.compileGLSL(hashTableSource, "compute");
  const resizeShaderCode = glslang.compileGLSL(resizeSource, "compute");
  resultsTextArea.value += "GLSL code compiled.\n";

  const hashTableShaderModule = device.createShaderModule({
    code: hashTableShaderCode,
  });
  const hashTableBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });
  const hashTablePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [hashTableBindGroupLayout],
    }),
    compute: {
      module: hashTableShaderModule,
      entryPoint: "main",
    },
  });

  const resizeShaderModule = device.createShaderModule({
    code: resizeShaderCode,
  });
  const resizeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  const resizePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [resizeBindGroupLayout],
    }),
    compute: {
      module: resizeShaderModule,
      entryPoint: "main",
    },
  });

  runButton.addEventListener("click", (e) => {
    const RESIZE_THRESHOLD = Number(resizeThresholdInput.value);
    const NEW_SIZE_FACTOR = Number(resizeFactorInput.value);
    const INITIAL_CAPACITY = Number(initialSizeInput.value);
    const OPS_PER_BATCH = Number(opsPerBatchInput.value);
    const ITERATIONS = Number(batchesInput.value);
    const KEY_RANGE = Number(keyRangeInput.value);
    benchmark(
      device,
      hashTablePipeline,
      hashTableBindGroupLayout,
      resizePipeline,
      resizeBindGroupLayout,
      RESIZE_THRESHOLD,
      INITIAL_CAPACITY,
      OPS_PER_BATCH,
      ITERATIONS,
      KEY_RANGE,
      NEW_SIZE_FACTOR
    );
  });
})();

const readGPUBuffer = async (device, gpuBuffer, bufferSize) => {
  const gpuReadBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const commandEncoder = device.createCommandEncoder();

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(gpuBuffer, 0, gpuReadBuffer, 0, bufferSize);

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  return new Int32Array(arrayBuffer);
};

const printHashtableBuffer = async (device, hashtableBuffer, hashtableBufferSize) => {
  const blocks = await readGPUBuffer(
    device,
    hashtableBuffer,
    hashtableSize * Int32Array.BYTES_PER_ELEMENT
  );
  console.log("Blocks:", blocks);
  const keys = [];
  for (let i = 0; i < blocks.length; ++i) {
    if (blocks[i] === 0) {
      continue;
    }
    keys.push(blocks[i]);
  }
  console.log("Keys:", keys);
  console.log("Unique keys:", new Set(keys));
};