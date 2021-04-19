#version 450
#define NIL 0u
#define TOMBSTONE 0xFFFFFFu

layout(std430, set = 0, binding = 0) readonly buffer metadataBuffer {
  int blockSize;
  int slabSize;
  int hashTableLength;
  volatile int padding[29];
}
metadata;

layout(std430, set = 0, binding = 1) buffer hashTableBuffer { uint pointers[]; }
hashTable;

layout(std430, set = 0, binding = 2) buffer allocaterMetadataBuffer {
  uint maxSlab;
  uint allocatedCounter;
  uint deallocatedCounter;
  uint reusedCounter;
  uint failedCounter;
  volatile int padding[27];
}
allocaterMetadata;

layout(std430, set = 0, binding = 3) buffer deallocatedListBuffer {
  uint data[];
}
deallocatedList;

layout(std430, set = 0, binding = 4) buffer blocksBuffer {
  coherent uint data[];
}
blocks;

layout(std430, set = 0, binding = 5) buffer oldBlocksBuffer {
  coherent uint data[];
}
oldBlocks;

uint murmur3(uint key) {
  const uint seed = 0x1a8b714cu;
  const uint c1 = 0xcc9e2d51u;
  const uint c2 = 0x1b873593u;
  const uint n = 0xe6546b64u;
  uint k = key;

  k = k * c1;
  k = (k << 15u) | (k >> 17u);
  k = k * c2;

  uint h = k ^ seed;
  h = (h << 13u) | (h >> 19u);
  h = h * 5u + n;
  h = h ^ 4u;

  h = h ^ (h >> 16u);
  h = h * 0x85ebca6bu;
  h = h ^ (h >> 13u);
  h = h * 0xc2b2ae35u;
  h = h ^ (h >> 16u);
  return h;
}

uint allocateSlab() {
  uint deallocatedCounter = allocaterMetadata.deallocatedCounter;
  uint reusedCounter = allocaterMetadata.reusedCounter;
  // try to find a reusable slab
  if (deallocatedCounter > reusedCounter) {
    uint maxIndex =
        min(deallocatedCounter, uint(metadata.blockSize / metadata.slabSize));
    for (uint i = 0; i < maxIndex; ++i) {
      uint currentBlock = deallocatedList.data[i];
      if (currentBlock != NIL) {
        uint compSwapResult =
            atomicCompSwap(deallocatedList.data[i], currentBlock, NIL);
        if (compSwapResult == currentBlock) {
          atomicAdd(allocaterMetadata.reusedCounter, 1);
          return currentBlock;
        }
      }
    }
  }
  // no reusable slab
  uint oldBlocks = atomicAdd(allocaterMetadata.allocatedCounter, 1);
  if (oldBlocks <= allocaterMetadata.maxSlab) {
    return oldBlocks * metadata.slabSize;
  } else {
    return NIL;
  }
}

bool deallocateSlab(uint slabAddress) {
  uint maxIndex = uint(metadata.blockSize / metadata.slabSize);
  for (uint i = 0; i < maxIndex; ++i) {
    uint compSwapResult =
        atomicCompSwap(deallocatedList.data[i], NIL, slabAddress);
    if (compSwapResult == NIL) {
      atomicAdd(allocaterMetadata.deallocatedCounter, 1);
      return true;
    }
  }
  return false;
}

bool insertIfAbsent(uint key) {
  uint h = murmur3(uint(key)) % metadata.hashTableLength;
  uint currentPointer = hashTable.pointers[h];
  // first insertion into the bucket
  while (currentPointer == NIL) {
    // try to read again with atomic read
    currentPointer = hashTable.pointers[h];
    if (currentPointer != NIL) {
      break;
    } else {
      uint newSlabPointer = allocateSlab();
      if (newSlabPointer == NIL) {
        continue;
      }
      uint compSwapResult =
          atomicCompSwap(hashTable.pointers[h], NIL, newSlabPointer);
      if (compSwapResult != NIL) {
        // someone else allocated a slab
        if (newSlabPointer != NIL) {
          deallocateSlab(newSlabPointer);
        }
        currentPointer = compSwapResult;
      } else {
        currentPointer = newSlabPointer;
      }
    }
  }
  while (true) {
    uint found = blocks.data[currentPointer];
    if (found == key) {
      return false;
    } else if (found == NIL) {
      uint compSwapResult =
          atomicCompSwap(blocks.data[currentPointer], NIL, key);
      if (compSwapResult == NIL) {
        return true;
      }
    }
    currentPointer += 1;
    if (currentPointer % metadata.slabSize == metadata.slabSize - 1) {
      // last entry of slab
      uint currentPointerCopy = currentPointer;
      currentPointer = blocks.data[currentPointerCopy];
      while (currentPointer == NIL) {
        // try to read again
        currentPointer = blocks.data[currentPointerCopy];
        if (currentPointer != NIL) {
          break;
        } else {
          uint newSlabPointer = allocateSlab();
          if (newSlabPointer != NIL) {
            uint compSwapResult = atomicCompSwap(
                blocks.data[currentPointerCopy], NIL, newSlabPointer);
            if (compSwapResult != NIL) {
              // someone else allocated a slab
              if (newSlabPointer != NIL) {
                currentPointer = compSwapResult;
                deallocateSlab(newSlabPointer);
              }
            } else {
              currentPointer = newSlabPointer;
            }
          }
        }
      }
    }
  }
  return false;
}

void main() {
  uint id = uint(gl_GlobalInvocationID.x);
  if ((id % metadata.slabSize) == (metadata.slabSize - 1)) {
    return;
  }
  uint currentKey = oldBlocks.data[id];
  if (currentKey == NIL || currentKey == TOMBSTONE) {
    return;
  }
  insertIfAbsent(currentKey);
}