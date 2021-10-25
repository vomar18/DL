start = time.time()
W = LSRegression(X, Y)
print(f"Elapsed: {time.time() - start:.6f} seconds")