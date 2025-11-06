-- Create clicks table
CREATE TABLE IF NOT EXISTS clicks (
    click_id SERIAL PRIMARY KEY,
    ip VARCHAR(50),
    app_id INTEGER,
    device_id INTEGER,
    os_id INTEGER,
    channel_id INTEGER,
    click_time TIMESTAMP,
    is_attributed INTEGER,
    fraud_score FLOAT,
    is_fraud BOOLEAN,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ip ON clicks(ip);
CREATE INDEX idx_click_time ON clicks(click_time);
CREATE INDEX idx_fraud_score ON clicks(fraud_score);

-- Create fraud_stats table
CREATE TABLE IF NOT EXISTS fraud_stats (
    id SERIAL PRIMARY KEY,
    date DATE,
    total_clicks INTEGER,
    fraud_clicks INTEGER,
    fraud_rate FLOAT,
    money_saved FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
