X_train, y_train = [], []
for i in range(train_labels.shape[0]):
    X_train.append(scaled_train_features.iloc[i:i+T].values)
    y_train.append(train_labels.iloc[i])
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,1)




for i in range(train_labels.shape[0]):
    X_train.append(scaled_train_features.iloc[i:i+T].values)
    y_train.append(train_labels.iloc[i])
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,1)  
print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')

X_train, y_train = [], []
for i in range(train_labels.shape[0]):
    X_train.append(scaled_train_features.iloc[i:i+T].values)
    y_train.append(train_labels.iloc[i])
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,1)  

print(f'train data dimensions: {X_train.shape}, {y_train.shape}')