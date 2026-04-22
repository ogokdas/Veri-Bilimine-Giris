import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def add_time_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    return df

train = add_time_features(train)
test = add_time_features(test)

features = [
    "t1", "t2", "hum", "wind_speed", "weather_code",
    "is_holiday", "is_weekend", "season",
    "hour", "dayofweek", "month"
]

X_train = train[features]
y_train = train["cnt"]
X_test = test[features]

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

# Bisiklet sayısı negatif olamayacağı için negatif tahminleri 0'a çekelim
pred = pred.clip(min=0)

submission = pd.DataFrame({
    "row_id": test["row_id"],
    "cnt": pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv oluşturuldu.")