#version 450
layout(std430, set = 0, binding = 0) readonly buffer metadataBuffer {
  int capacity;
  uint testBatch;
  uint keyRange;
  uint NIL;
  uint TOMBSTONE;
  volatile int padding[26];
}
metadata;

layout(std430, set = 0, binding = 1) buffer countersBuffer {
  uint counterInsertions;
  uint counterDeletions;
  volatile int padding[30];
}
counters;

layout(std430, set = 0, binding = 2) buffer hashTableBuffer {
  uint data[];
}
hashTable;

// Pseudo random generator:
// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint randomHash(uint x) {
  x += (x << 10u);
  x ^= (x >> 6u);
  x += (x << 3u);
  x ^= (x >> 11u);
  x += (x << 15u);
  return x;
}

uint randomHash(uint x, uint y) { return randomHash(x ^ randomHash(y)); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value
// below 1.0.
float floatConstruct(uint m) {
  const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
  const uint ieeeOne = 0x3F800000u;      // 1.0 in IEEE binary32

  m &= ieeeMantissa; // Keep only mantissa bits (fractional part)
  m |= ieeeOne;      // Add fractional part to 1.0

  float f = uintBitsToFloat(m); // Range [1:2]
  return f - 1.0;               // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random(uint x) { return floatConstruct(randomHash(x)); }
float random(uint x, uint y) { return floatConstruct(randomHash(x, y)); }

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

bool isKeyExist(uint key) {
  uint h = murmur3(uint(key));
  for (uint i = 0; i < metadata.capacity; ++i) {
    uint index = (h + i) % metadata.capacity;
    uint found = hashTable.data[index];
    if (found == key) {
      return true;
    } else if (found == metadata.NIL) {
      return false;
    }
  }
  return false;
}

bool insertIfAbsent(uint key) {
  uint h = murmur3(uint(key));
  for (uint i = 0; i < metadata.capacity; ++i) {
    uint index = (h + i) % metadata.capacity;
    uint found = hashTable.data[index];
    if (found == key) {
      return false;
    } else if (found == metadata.NIL) {
      uint compSwapResult =
          atomicCompSwap(hashTable.data[index], metadata.NIL, key);
      if (compSwapResult == metadata.NIL) {
        atomicAdd(counters.counterInsertions, 1);
        return true;
      } else if (compSwapResult == key) {
        return false;
      }
    }
  }
  return false;
}

bool erase(uint key) {
  uint h = murmur3(uint(key));
  for (uint i = 0; i < metadata.capacity; ++i) {
    uint index = (h + i) % metadata.capacity;
    uint found = hashTable.data[index];
    if (found == metadata.NIL) {
      return false;
    } else if (found == key) {
      uint compSwapResult =
          atomicCompSwap(hashTable.data[index], key, metadata.TOMBSTONE);
      if (compSwapResult == key) {
        atomicAdd(counters.counterDeletions, 1);
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}

void main() {
  uint id = uint(gl_GlobalInvocationID.x) + 1u;
  float rand = random(metadata.testBatch, id);
  float rand2 = random(id, metadata.testBatch);
  uint key = uint(rand * metadata.keyRange);

  if (rand2 > 0.5) {
    insertIfAbsent(key);
  } else {
    erase(key);
  }
}