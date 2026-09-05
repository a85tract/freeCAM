// The document's hash, computed the same way the Python side computes it:
// compact JSON with keys sorted by code point, numbers as JavaScript writes
// them, strings with their characters intact, over the execution record.
// One document, one hash, wherever it is computed.

import type { NodeConfiguration, WorkflowDocument, WorkflowNode } from "./types";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export function canonical(value: Json): string {
  if (value === null || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("a workflow document holds finite numbers only");
    return String(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  const keys = Object.keys(value).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return "{" + keys.map((key) => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}";
}

/** The configuration as the hash sees it: only what changes execution. */
export function canonicalConfiguration(configuration: NodeConfiguration): Json {
  const kernels: { [key: string]: Json } = {};
  for (const name of Object.keys(configuration.kernels).sort()) {
    const binding = configuration.kernels[name];
    if (binding.kind !== "original") kernels[name] = { kind: binding.kind, path: binding.path };
  }
  return {
    parameters: configuration.parameters as Json,
    python_source: configuration.python_source,
    kernels,
    variables: configuration.variables.map((v) => ({ name: v.name, like: v.like, units: v.units, output: v.output })),
  };
}

export function executionRecord(document: Pick<WorkflowDocument, "case" | "nsteps" | "namelist" | "nodes">): Json {
  return {
    case: document.case,
    nsteps: document.nsteps,
    namelist: document.namelist as Json,
    nodes: document.nodes.map((node: WorkflowNode) => [
      node.id,
      node.qualified_name,
      node.origin,
      Boolean(node.enabled),
      canonicalConfiguration(node.configuration),
    ]),
  };
}

export function workflowHash(document: Pick<WorkflowDocument, "case" | "nsteps" | "namelist" | "nodes">): string {
  return sha256Hex(canonical(executionRecord(document)));
}

export function orderHash(document: Pick<WorkflowDocument, "nodes">): string {
  return sha256Hex(canonical(document.nodes.map((node) => [node.id, Boolean(node.enabled)])));
}

// --- SHA-256, synchronous, for strings (UTF-8) -------------------------------

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export function sha256Hex(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const length = bytes.length;
  const withPadding = ((length + 9 + 63) >> 6) << 6;
  const buffer = new Uint8Array(withPadding);
  buffer.set(bytes);
  buffer[length] = 0x80;
  const view = new DataView(buffer.buffer);
  const bits = length * 8;
  view.setUint32(withPadding - 8, Math.floor(bits / 0x100000000));
  view.setUint32(withPadding - 4, bits >>> 0);

  const h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const w = new Uint32Array(64);
  for (let offset = 0; offset < withPadding; offset += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(offset + i * 4);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, hh] = h;
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + K[i] + w[i]) >>> 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] = (h[0] + a) >>> 0; h[1] = (h[1] + b) >>> 0; h[2] = (h[2] + c) >>> 0; h[3] = (h[3] + d) >>> 0;
    h[4] = (h[4] + e) >>> 0; h[5] = (h[5] + f) >>> 0; h[6] = (h[6] + g) >>> 0; h[7] = (h[7] + hh) >>> 0;
  }
  return Array.from(h, (word) => word.toString(16).padStart(8, "0")).join("");
}

function rotr(value: number, bits: number): number {
  return (value >>> bits) | (value << (32 - bits));
}
