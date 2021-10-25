# la tua implementazione del GD è giusta se e solo se gli errori sono uguali
# Standardizing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
