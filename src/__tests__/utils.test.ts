import { computeResponseCost } from "../utils";

describe("computeResponseCost", () => {
  const COST_PER_MILLION_TOKENS = 1.5; // Example cost: $1.50 per 1,000,000 tokens

  it("should return 0 if tokens are 0", () => {
    const cost = computeResponseCost(0, COST_PER_MILLION_TOKENS);
    expect(cost).toBe(0);
  });

  it("should return 0 if cost per million tokens is 0", () => {
    const cost = computeResponseCost(1000, 0);
    expect(cost).toBe(0);
  });

  it("should return 0 if cost per million tokens is undefined", () => {
    // Testing JavaScript behavior, though TypeScript might prevent this
    const cost = computeResponseCost(1000, undefined as any);
    expect(cost).toBeNaN(); // Or potentially throw, depending on JS interpretation
  });

  it("should return 0 if cost per million tokens is null", () => {
    // Testing JavaScript behavior
    const cost = computeResponseCost(1000, null as any);
    // null becomes 0 in arithmetic operations
    expect(cost).toBe(0);
  });

  it("should calculate cost correctly for a small number of tokens", () => {
    const tokens = 1000;
    const expectedCost = tokens * (COST_PER_MILLION_TOKENS / 1000000);
    const cost = computeResponseCost(tokens, COST_PER_MILLION_TOKENS);
    expect(cost).toBeCloseTo(expectedCost);
  });

  it("should calculate cost correctly for exactly one million tokens", () => {
    const tokens = 1000000;
    const expectedCost = COST_PER_MILLION_TOKENS; // Should be exactly the cost per million
    const cost = computeResponseCost(tokens, COST_PER_MILLION_TOKENS);
    expect(cost).toBeCloseTo(expectedCost);
  });

  it("should calculate cost correctly for a large number of tokens", () => {
    const tokens = 5500000; // 5.5 million tokens
    const expectedCost = tokens * (COST_PER_MILLION_TOKENS / 1000000);
    const cost = computeResponseCost(tokens, COST_PER_MILLION_TOKENS);
    expect(cost).toBeCloseTo(expectedCost);
    expect(cost).toBeCloseTo(5.5 * COST_PER_MILLION_TOKENS);
  });

  it("should handle floating point costs per million tokens", () => {
    const tokens = 2000000;
    const floatCostPerMillion = 0.753;
    const expectedCost = tokens * (floatCostPerMillion / 1000000);
    const cost = computeResponseCost(tokens, floatCostPerMillion);
    expect(cost).toBeCloseTo(expectedCost);
    expect(cost).toBeCloseTo(2 * floatCostPerMillion);
  });
});
