# Import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    def __init__(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    def scaler_fit_transform(self, X):
        X_train_flat = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_train_flat)
        X_scaled = self.scaler.transform(X_train_flat).reshape(X.shape)
        return X_scaled
        
    def scaler_transform(self, X):
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_flat).reshape(X.shape)
        return X_scaled
      
    def encoder_fit_transform(self, y):
        return self.encoder.fit_transform(y.reshape(-1, 1))
        
    def encoder_transform(self, y):
        return self.encoder.transform(y.reshape(-1, 1))