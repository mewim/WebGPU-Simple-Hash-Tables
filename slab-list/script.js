const initialCapacityInput = document.getElementById("initial-capacity");
const hashTableLengthInput = document.getElementById("hash-table-length");
const slabSizeInput = document.getElementById("slab-size");
const resizeFactorInput = document.getElementById("resize-factor");
const resizeThresholdInput = document.getElementById("resize-threshold");
const keyRangeInput = document.getElementById("key-range");
const batchesInput = document.getElementById("batches");
const opsPerBatchInput = document.getElementById("ops-per-batch");
const runButton = document.getElementById("run");
const resultsTextArea = document.getElementById("results");

// The block size to assure `capacity` insertions can be performed successfully
const computeBlockSize = (capacity, slabSize, hashTableLength) => {
  return (
    Math.ceil(capacity / (slabSize - 1)) * slabSize + // space for desired capacity
    hashTableLength * slabSize + // extra space if the last slab is under-utilized
    slabSize // extra padding at the front because the first slab cannot be allocated
  );
};

const benchmark = async (
  device,
  hashTablePipeline,
  hashTableBindGroupLayout,
  resizePipeline,
  resizeBindGroupLayout,
  HASH_TABLE_LENGTH,
  SLAB_SIZE,
  RESIZE_THRESHOLD,
  INITIAL_CAPACITY,
  OPS_PER_BATCH,
  ITERATIONS,
  KEY_RANGE,
  NEW_SIZE_FACTOR
) => {
  resultsTextArea.value += `Configurations: \n`;
  resultsTextArea.value += `\tINITIAL_CAPACITY=${INITIAL_CAPACITY} \n`;
  resultsTextArea.value += `\tHASH_TABLE_LENGTH=${HASH_TABLE_LENGTH} \n`;
  resultsTextArea.value += `\tSLAB_SIZE=${SLAB_SIZE} \n`;
  resultsTextArea.value += `\tRESIZE_THRESHOLD=${RESIZE_THRESHOLD} \n`;
  resultsTextArea.value += `\tOPS_PER_BATCH=${OPS_PER_BATCH} \n`;
  resultsTextArea.value += `\tITERATIONS=${ITERATIONS} \n`;
  resultsTextArea.value += `\tKEY_RANGE=${KEY_RANGE} \n`;
  resultsTextArea.value += `\tNEW_SIZE_FACTOR=${NEW_SIZE_FACTOR} \n`;

  let hashTableLength = HASH_TABLE_LENGTH;
  let blockSize = computeBlockSize(
    INITIAL_CAPACITY,
    SLAB_SIZE,
    hashTableLength
  );
  resultsTextArea.value += `\tBLOCK_SIZE=${blockSize} \n`;

  let currentCapacity = INITIAL_CAPACITY;
  let totalInsertions = 0;
  let totalDeletions = 0;
  let {
    deallocatedListBuffer,
    blocksBuffer,
    hashTableBuffer,
    allocaterMetadataBuffer,
  } = createBuffersForHashTable(device, hashTableLength, blockSize, SLAB_SIZE);

  for (let i = 0; i < ITERATIONS; ++i) {
    const startTime = performance.now();
    if (totalInsertions >= currentCapacity * RESIZE_THRESHOLD) {
      let newCapacity = (totalInsertions - totalDeletions) * NEW_SIZE_FACTOR;
      if (newCapacity < OPS_PER_BATCH) {
        newCapacity = OPS_PER_BATCH;
      }
      const resizeStartTime = performance.now();
      deallocatedListBuffer.destroy();
      hashTableBuffer.destroy();
      allocaterMetadataBuffer.destroy();

      const resizedHashTable = await runResizeCode(
        device,
        resizeBindGroupLayout,
        resizePipeline,
        blocksBuffer,
        blockSize,
        currentCapacity,
        hashTableLength,
        SLAB_SIZE,
        newCapacity
      );
      deallocatedListBuffer = resizedHashTable.deallocatedListBuffer;
      hashTableBuffer = resizedHashTable.hashTableBuffer;
      blocksBuffer = resizedHashTable.blocksBuffer;
      allocaterMetadataBuffer = resizedHashTable.allocaterMetadataBuffer;

      const resizeTime = performance.now() - resizeStartTime;
      resultsTextArea.value += `Iteration ${
        i + 1
      }:\tresize completed in ${resizeTime.toFixed(
        2
      )} ms. Old capacity: ${currentCapacity},\tnew capacity: ${newCapacity},\tnew hash table length: ${
        resizedHashTable.newHashTableLength
      },\tnew block size: ${resizedHashTable.newBlockSize}.\n`;

      currentCapacity = newCapacity;
      blockSize = resizedHashTable.newBlockSize;
      hashTableLength = resizedHashTable.newHashTableLength;
      totalInsertions = totalInsertions - totalDeletions;
      totalDeletions = 0;
    }
    const counters = await runHashTableCode(
      device,
      hashTableBindGroupLayout,
      hashTableBuffer,
      allocaterMetadataBuffer,
      deallocatedListBuffer,
      blocksBuffer,
      hashTablePipeline,
      blockSize,
      SLAB_SIZE,
      hashTableLength,
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

const createAllocatorBuffer = (device, blockSize, slabSize) => {
  const allocaterMetadataSize = 32;
  const allocaterMetadataArray = new Int32Array(allocaterMetadataSize);
  allocaterMetadataArray[0] = (blockSize - slabSize) / slabSize; // max slab index
  allocaterMetadataArray[1] = 1; // disable first slab

  const allocaterMetadataBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: allocaterMetadataSize * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const mappedAllocaterMetadataBuffer = allocaterMetadataBuffer.getMappedRange();
  new Int32Array(mappedAllocaterMetadataBuffer).set(allocaterMetadataArray);
  allocaterMetadataBuffer.unmap();
  return allocaterMetadataBuffer;
};

const createBuffersForHashTable = (
  device,
  hashTableLength,
  blockSize,
  slabSize
) => {
  let deallocatedListBuffer = device.createBuffer({
    size: (blockSize / slabSize) * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  let blocksBuffer = device.createBuffer({
    size: blockSize * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  let hashTableBuffer = device.createBuffer({
    size: hashTableLength * Int32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  let allocaterMetadataBuffer = createAllocatorBuffer(
    device,
    blockSize,
    slabSize
  );
  return {
    deallocatedListBuffer,
    blocksBuffer,
    hashTableBuffer,
    allocaterMetadataBuffer,
  };
};

const runHashTableCode = async (
  device,
  bindGroupLayout,
  hashTableBuffer,
  allocaterMetadataBuffer,
  deallocatedListBuffer,
  blocksBuffer,
  hashTablePipeline,
  blockSize,
  slabSize,
  hashTableLength,
  currentIteration,
  keyRange,
  numberOfOps
) => {
  const metadataSize = 32;
  const metadataArray = new Int32Array(metadataSize);
  metadataArray[0] = blockSize;
  metadataArray[1] = slabSize;
  metadataArray[2] = hashTableLength;
  metadataArray[3] = currentIteration;
  metadataArray[4] = keyRange;

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
      {
        binding: 3,
        resource: {
          buffer: allocaterMetadataBuffer,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: deallocatedListBuffer,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: blocksBuffer,
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
  const counters = await readGPUBuffer(device, countersBuffer, 8);
  countersBuffer.destroy();

  return counters;
};

const runResizeCode = async (
  device,
  bindGroupLayout,
  resizePipeline,
  oldBlocksBuffer,
  oldBlocksSize,
  oldCapacity,
  oldHashTableLength,
  slabSize,
  newCapacity
) => {
  // Make sure the length of hash table changes with capacity
  const newHashTableLength =
    Math.ceil(newCapacity / oldCapacity * oldHashTableLength);
  console.log(newHashTableLength);

  const newBlockSize = computeBlockSize(
    newCapacity,
    slabSize,
    newHashTableLength
  );

  let {
    deallocatedListBuffer,
    blocksBuffer,
    hashTableBuffer,
    allocaterMetadataBuffer,
  } = createBuffersForHashTable(
    device,
    newHashTableLength,
    newBlockSize,
    slabSize
  );

  const metadataSize = 32;
  const metadataArray = new Int32Array(metadataSize);
  metadataArray[0] = newBlockSize;
  metadataArray[1] = slabSize;
  metadataArray[2] = newHashTableLength;
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
          buffer: hashTableBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: allocaterMetadataBuffer,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: deallocatedListBuffer,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: blocksBuffer,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: oldBlocksBuffer,
        },
      },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(resizePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(oldBlocksSize);
  passEncoder.endPass();
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);
  await device.queue.onSubmittedWorkDone();
  oldBlocksBuffer.destroy();
  metadataBuffer.destroy();

  return {
    hashTableBuffer,
    allocaterMetadataBuffer,
    deallocatedListBuffer,
    blocksBuffer,
    newBlockSize,
    newHashTableLength,
    newCapacity,
  };
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
  const bindGroupLayout = {
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
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  };
  const hashTableBindGroupLayout = device.createBindGroupLayout(
    bindGroupLayout
  );
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
  const resizeBindGroupLayout = device.createBindGroupLayout(bindGroupLayout);

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
    const HASH_TABLE_LENGTH = Number(hashTableLengthInput.value);
    const SLAB_SIZE = Number(slabSizeInput.value);
    const RESIZE_THRESHOLD = Number(resizeThresholdInput.value);
    const INITIAL_CAPACITY = Number(initialCapacityInput.value);
    const OPS_PER_BATCH = Number(opsPerBatchInput.value);
    const ITERATIONS = Number(batchesInput.value);
    const KEY_RANGE = Number(keyRangeInput.value);
    const NEW_SIZE_FACTOR = Number(resizeFactorInput.value);

    benchmark(
      device,
      hashTablePipeline,
      hashTableBindGroupLayout,
      resizePipeline,
      resizeBindGroupLayout,
      HASH_TABLE_LENGTH,
      SLAB_SIZE,
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
