"""

Stock Predictions – HackerRank AI Challenge

This Python bot solves the "Stock Predictions" one-player game on HackerRank.

We start with $100. Each day we're given k stocks, each with the last 5 days of prices (oldest to latest), how many shares we own, and how many days are left.

The goal: issue BUY and SELL orders daily to maximize your total money by the final day. Money from sells is only available the next day.

Strategy:

1. SELL all stocks that went up today (p4 > p3)
2. For all stocks that dropped today, calculate % drop and store them in a max-heap
3. BUY the biggest losers first by popping from the heap and spending all available cash greedily

This simple mean-reversion approach, powered by a max-heap prioritization of daily losers, consistently earns over $80,000 in the hidden test set — scoring above 56 points (well over the 40-point challenge threshold).

Note:
Various more advanced indicators were tested (moving averages, Bollinger bands, RSI, etc.), but none outperformed this simple greedy mean-reversion strategy.

---

Example Input:

90 2 400
iStreet 10 4.54 5.53 6.56 5.54 7.60
HR 0 30.54 27.53 24.42 20.11 17.50

Explanation:
- 90: Available cash today
- 2: Number of stocks
- 400: Days remaining
- iStreet: Own 10 shares, price rose today (5.54 -> 7.60) -> SELL
- HR: Dropped significantly today (20.11 -> 17.50) -> BUY 5 shares

Expected Output:
2
iStreet SELL 10
HR BUY 5


"""

import sys, heapq

# Main trading function
def printTransactions(m, k, d, names, owned, prices):
    tx = []

    # Step 1: Sell stocks that went up today (lock in profit)
    for i in range(k):
        if owned[i] and prices[i][4] > prices[i][3]:
            tx.append(f"{names[i]} SELL {owned[i]}")

    # Step 2: Build a max-heap of biggest price drops from yesterday
    heap = []
    for i in range(k):
        cur, prev = prices[i][4], prices[i][3]
        if cur < prev:
            drop = (prev - cur) / cur
            if drop > 0:
                heapq.heappush(heap, (-drop, i))  

    # Step 3: Buy the biggest losers using available cash
    cash = m
    while heap and cash > 0:
        _, i = heapq.heappop(heap)
        price = prices[i][4]
        if cash < price:
            break
        qty = int(cash // price)
        if qty and qty * price < 1e8:
            tx.append(f"{names[i]} BUY {qty}")
            cash -= qty * price

    # Output the number of transactions followed by each transaction
    print(len(tx))
    for t in tx:
        print(t)

# HackerRank input parsing
if __name__ == "__main__":
    head = sys.stdin.readline().split()
    if not head:
        sys.exit()
    m, k, d = float(head[0]), int(head[1]), int(head[2])
    names, owned, prices = [], [], []
    for _ in range(k):
        row = sys.stdin.readline().split()
        names.append(row[0])
        owned.append(int(row[1]))
        prices.append(list(map(float, row[2:])))
    printTransactions(m, k, d, names, owned, prices)
