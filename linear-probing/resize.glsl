#version 450
layout(std430, set = 0, binding = 0) readonly buffer metadataBuffer {
  uint oldCapacity;
  uint newCapacity;
  uint NIL;
  uint TOMBSTONE;
  int padding[28];
}
metadata;

layout(std430, set = 0, binding = 1) buffer oldHashTableBuffer { uint data[]; }
oldHashTable;

layout(std430, set = 0, binding = 2) buffer newHashTableBuffer { uint data[]; }
newHashTable;

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

bool insertIfAbsent(uint key) {
  uint h = murmur3(uint(key));
  for (uint i = 0; i < metadata.newCapacity; ++i) {
    uint index = (h + i) % metadata.newCapacity;
    uint found = newHashTable.data[index];
    if (found == key) {
      return false;
    } else if (found == metadata.NIL) {
      uint compSwapResult =
          atomicCompSwap(newHashTable.data[index], metadata.NIL, key);
      if (compSwapResult == metadata.NIL) {
        return true;
      } else if (compSwapResult == key) {
        return false;
      }
    }
  }
  return false;
}

void main() {
  uint id = uint(gl_GlobalInvocationID.x);
  uint currentKey = oldHashTable.data[id];
  if (currentKey == metadata.NIL || currentKey == metadata.TOMBSTONE) {
    return;
  }
  insertIfAbsent(currentKey);
}