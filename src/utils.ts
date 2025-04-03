export function computeResponseCost (tokens: number, cost: number): number {
    const tokenCost = cost / 1000000
    return tokens * tokenCost;
}