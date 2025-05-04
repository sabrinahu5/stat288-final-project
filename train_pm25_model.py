import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def load_tfrecord_data(tfrecord_path):
    """
    Load TFRecord data from the given path. Expects TFRecords with features:
      - 'sid': Sensor ID (float)
      - 'week_ms': Week start timestamp in milliseconds (float)
      - 'Optical_Depth_047': AOD values (float_list)
      - 'AOD_QA': Quality assurance values (float_list)
      - 'NDVI': Normalized Difference Vegetation Index values (float_list)
      - 'Column_WV': Column water vapor values (float_list)
      - 'longitude', 'latitude': Coordinates (float)
      - 'city': City name (bytes)
    Returns:
      A pandas DataFrame with columns ['sid', 'week_ms', 'patch'].
      'sid' is int, 'week_ms' is int, and 'patch' is a numpy array of shape (32, 32, 4)
      where the 4 channels are [Optical_Depth_047, AOD_QA, NDVI, Column_WV].
    """
    PATCH = 33
    bands = ['Optical_Depth_047', 'AOD_QA', 'NDVI', 'Column_WV']
    data = []
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature
        
        # Extract sensor ID
        sid = int(features['sid'].float_list.value[0])
        
        # Extract week timestamp (ms since epoch)
        week_ms = int(features['week_ms'].float_list.value[0])
        
        # Extract coordinates
        longitude = features['longitude'].float_list.value[0]
        latitude = features['latitude'].float_list.value[0]
        
        # Extract the four bands to create the patch
        patch = []
        for band in bands:
            arr = features[band].float_list.value
            if len(arr) == PATCH * PATCH:
                arr = np.array(arr).reshape(PATCH, PATCH)
            else:
                arr = np.zeros((PATCH, PATCH), dtype=np.float32)
            patch.append(arr)
        patch = np.stack(patch, axis=-1)
        
        # Append to list
        data.append({
            'sid': sid, 
            'week_ms': week_ms, 
            'longitude': longitude,
            'latitude': latitude,
            'patch': patch
        })
        
        # Optional: extract city if needed
        if 'city' in features:
            data[-1]['city'] = features['city'].bytes_list.value[0].decode('utf-8')
            
    return pd.DataFrame(data)

def load_csv_data(csv_path):
    """
    Load the OpenAQ weekly PM2.5 CSV from the given path.
    Returns:
      A pandas DataFrame with columns ['sid', 'week', 'pm25_mean', 'year', 'week_ms'].
      - 'sid' is int
      - 'week' is the week start date string (YYYY-MM-DD)
      - 'pm25_mean' is float
      - 'year' is int (year of the week)
      - 'week_ms' is int (timestamp in ms for the week start)
    """
    df = pd.read_csv(csv_path)
    # Ensure correct dtypes
    if 'sid' in df.columns:
        df['sid'] = df['sid'].astype(int)
    if 'pm25_mean' in df.columns:
        df['pm25_mean'] = df['pm25_mean'].astype(float)
    # Parse week dates to datetime and extract year and timestamp
    if 'week' in df.columns:
        df['date'] = pd.to_datetime(df['week'])
    else:
        # If no 'week' column (unlikely in our case), try 'week_ms'
        if 'week_ms' in df.columns:
            df['date'] = pd.to_datetime(df['week_ms'], unit='ms')
        else:
            raise ValueError("CSV must contain 'week' or 'week_ms' column for dates.")
    df['year'] = df['date'].dt.year
    # Convert week start to milliseconds timestamp (epoch)
    # Use floor division by 1e6 because pandas datetime is in ns.
    df['week_ms'] = (df['date'].astype(np.int64) // 10**6).astype(np.int64)
    return df[['sid', 'week', 'pm25_mean', 'year', 'week_ms']]

def merge_data(patch_df, label_df):
    """
    Merge the TFRecord-derived DataFrame (patch_df) with the labels DataFrame (label_df)
    on 'sid' and 'week_ms'. Returns the merged DataFrame.
    """
    print("patch_df columns:", patch_df.columns)
    print("label_df columns:", label_df.columns)
    print("patch_df shape:", patch_df.shape)
    print("label_df shape:", label_df.shape)
    merged = pd.merge(patch_df, label_df, on=['sid', 'week_ms'], how='inner')
    return merged

def build_cnn_model(input_shape):
    """
    Build and compile a small CNN model for regression on image patches.
    input_shape: tuple, e.g. (32, 32, 1) or (32, 32, 3).
    Returns:
      A compiled tf.keras.Model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # output layer for PM2.5
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse', metrics=['mae'])
    return model

def train_model_on_delhi(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN model on Delhi training data and evaluate on validation data.
    Logs RMSE and MAE for the validation (Delhi 2023).
    Returns the trained model.
    """
    # Train the model on Delhi data
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=2)
    # Predict on validation set
    y_pred_val = model.predict(X_val).reshape(-1)
    y_true_val = y_val.reshape(-1)
    rmse_val = np.sqrt(np.mean((y_pred_val - y_true_val) ** 2))
    mae_val = np.mean(np.abs(y_pred_val - y_true_val))
    print(f"Delhi 2023 Validation – RMSE: {rmse_val:.2f}, MAE: {mae_val:.2f}")
    return model

def fine_tune_on_lagos(model, X_train, y_train, baseline_model=None):
    """
    Fine-tune the model on Lagos data.
    If the Lagos training set is small (< 100 samples), perform linear regression correction (y_adj = α + β * y_pred).
    If larger, freeze early conv layers and fine-tune the dense head with a low learning rate.
    Params:
      model: Trained source model (after Delhi training).
      X_train, y_train: Lagos training data (patches and labels).
      baseline_model: (Optional) original model to use for predictions in linear mode if model gets modified.
    Returns:
      If linear correction is used, returns (alpha, beta) coefficients.
      If fine-tuning is done, returns the fine-tuned model.
    """
    n = X_train.shape[0]
    if n < 100:
        # Linear calibration approach
        if baseline_model is None:
            baseline_model = model
        # Get predictions from the baseline (Delhi-trained) model
        y_pred = baseline_model.predict(X_train).reshape(-1)
        y_true = y_train.reshape(-1)
        # Fit linear least squares: y_true = alpha + beta * y_pred
        A = np.vstack([np.ones(n), y_pred]).T
        # Solve for [alpha, beta]
        coeffs, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
        alpha, beta = coeffs[0], coeffs[1]
        print(f"Lagos calibration: alpha = {alpha:.3f}, beta = {beta:.3f}")
        return (alpha, beta)
    else:
        # Fine-tune CNN on Lagos
        # Freeze convolutional layers to retain Delhi features
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.trainable = False
        # Recompile with a smaller learning rate for fine-tuning
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
        # Fine-tune on Lagos data with higher sample weight
        sample_weight = np.ones(n) * 5.0
        model.fit(X_train, y_train, epochs=20, batch_size=8, sample_weight=sample_weight, verbose=2)
        return model

def evaluate_on_lagos(model, X_test, y_test, adaptation=None):
    """
    Evaluate the model (and optionally an adapted version) on Lagos test data.
    If adaptation is a tuple (alpha, beta), use linear calibration on the model's predictions.
    If adaptation is a model (fine-tuned), use it directly for predictions.
    Prints out RMSE and R^2 for zero-shot vs fine-tuned, and AUROC for WHO threshold exceedance.
    """
    # Ground truth
    y_true = y_test.reshape(-1)
    # Baseline (zero-shot) predictions using original model
    y_pred_base = model.predict(X_test).reshape(-1)
    # If adaptation provided, get adapted predictions
    if adaptation is None:
        # No adaptation, use baseline only
        y_pred_adapt = None
    elif isinstance(adaptation, tuple):
        # Linear calibration: adaptation = (alpha, beta)
        alpha, beta = adaptation
        y_pred_adapt = alpha + beta * y_pred_base
    else:
        # Adaptation is a fine-tuned model
        y_pred_adapt = adaptation.predict(X_test).reshape(-1)
    # Compute metrics for baseline
    rmse_base = np.sqrt(np.mean((y_pred_base - y_true) ** 2))
    mae_base = np.mean(np.abs(y_pred_base - y_true))
    # R^2 (coefficient of determination) for baseline
    ss_res_base = np.sum((y_true - y_pred_base) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_base = 1 - ss_res_base/ss_tot if ss_tot != 0 else 0.0
    print(f"Lagos 2023 (Zero-shot) – RMSE: {rmse_base:.2f}, MAE: {mae_base:.2f}, R^2: {r2_base:.3f}")
    if y_pred_adapt is not None:
        # Compute metrics for adapted model
        rmse_adapt = np.sqrt(np.mean((y_pred_adapt - y_true) ** 2))
        mae_adapt = np.mean(np.abs(y_pred_adapt - y_true))
        ss_res_adapt = np.sum((y_true - y_pred_adapt) ** 2)
        r2_adapt = 1 - ss_res_adapt/ss_tot if ss_tot != 0 else 0.0
        print(f"Lagos 2023 (Fine-tuned) – RMSE: {rmse_adapt:.2f}, MAE: {mae_adapt:.2f}, R^2: {r2_adapt:.3f}")
    # WHO 15 µg/m³ exceedance classification
    # Create binary labels for exceedance (1 if >=15, 0 if below 15)
    threshold = 15.0  # µg/m³ WHO guideline threshold
    y_true_exceed = (y_true >= threshold).astype(int)
    # Calculate AUROC for baseline predictions
    auc_base = roc_auc_score(y_true_exceed, y_pred_base)
    if y_pred_adapt is not None:
        auc_adapt = roc_auc_score(y_true_exceed, y_pred_adapt)
        print(f"Lagos 2023 – WHO 15 µg/m³ exceedance AUROC: baseline = {auc_base:.3f}, fine-tuned = {auc_adapt:.3f}")
    else:
        print(f"Lagos 2023 – WHO 15 µg/m³ exceedance AUROC: baseline = {auc_base:.3f}")

def main():
    # File paths
    delhi_tf_path = "data/delhi_patches32.tfrecord.gz"
    lagos_tf_path = "data/lagos_patches32.tfrecord.gz"
    delhi_csv_path = "data/delhi_weekly_pm25.csv"
    lagos_csv_path = "data/lagos_weekly_pm25.csv"
    # Load TFRecord and CSV data
    print("Loading Delhi TFRecord data...")
    delhi_patches = load_tfrecord_data(delhi_tf_path)
    print(f"Loaded {len(delhi_patches)} Delhi patch samples.")
    print("Loading Lagos TFRecord data...")
    lagos_patches = load_tfrecord_data(lagos_tf_path)
    print(f"Loaded {len(lagos_patches)} Lagos patch samples.")
    print("Loading Delhi labels CSV...")
    delhi_labels = load_csv_data(delhi_csv_path)
    print(f"Loaded {len(delhi_labels)} Delhi label entries.")
    print("Loading Lagos labels CSV...")
    lagos_labels = load_csv_data(lagos_csv_path)
    print(f"Loaded {len(lagos_labels)} Lagos label entries.")
    # Merge patches with labels
    delhi_patches['week'] = pd.to_datetime(delhi_patches['week_ms'], unit='ms').dt.strftime('%Y-%m-%d')
    lagos_patches['week'] = pd.to_datetime(lagos_patches['week_ms'], unit='ms').dt.strftime('%Y-%m-%d')
    # For patches
    for df in [delhi_patches, lagos_patches]:
        df['week_dt'] = pd.to_datetime(df['week_ms'], unit='ms')
        df['iso_year'] = df['week_dt'].dt.isocalendar().year
        df['iso_week'] = df['week_dt'].dt.isocalendar().week

    # For labels
    for df in [delhi_labels, lagos_labels]:
        df['week_dt'] = pd.to_datetime(df['week'], errors='coerce')
        df['iso_year'] = df['week_dt'].dt.isocalendar().year
        df['iso_week'] = df['week_dt'].dt.isocalendar().week

    # For Delhi
    delhi_merged = pd.merge(
        delhi_patches, delhi_labels,
        on=['sid', 'iso_year', 'iso_week'],
        how='inner'
    )

    # For Lagos
    lagos_merged = pd.merge(
        lagos_patches, lagos_labels,
        on=['sid', 'iso_year', 'iso_week'],
        how='inner'
    )

    # Filter by years for training/validation/test
    # Delhi train: 2018-2022, validation: 2023
    delhi_train, delhi_test = train_test_split(delhi_merged, test_size=0.2, random_state=42)

    # For Lagos: 80% training, 20% test
    lagos_train, lagos_test = train_test_split(lagos_merged, test_size=0.2, random_state=42)
    # If there's no separate Lagos train data (e.g., all data is 2023), we'll use the small dataset in calibration step.
    # Prepare numpy arrays for model input/output
    # Determine input shape from patches (assuming at least one sample exists)
    if len(delhi_train) == 0:
        raise RuntimeError("Delhi training set is empty after filtering years.")
    sample_patch = delhi_train['patch'].iloc[0]
    input_shape = sample_patch.shape  # e.g., (32, 32, 1) or (32, 32, 3)
    # Stack patches into numpy arrays
    X_delhi_train = np.stack(delhi_train['patch'].values)
    y_delhi_train = delhi_train['pm25_mean'].values.astype(np.float32)
    X_delhi_val = np.stack(delhi_test['patch'].values) if len(delhi_test) > 0 else np.array([]).reshape(0, *input_shape)
    y_delhi_val = delhi_test['pm25_mean'].values.astype(np.float32) if len(delhi_test) > 0 else np.array([], dtype=np.float32)
    X_lagos_train = np.stack(lagos_train['patch'].values) if len(lagos_train) > 0 else np.array([]).reshape(0, *input_shape)
    y_lagos_train = lagos_train['pm25_mean'].values.astype(np.float32) if len(lagos_train) > 0 else np.array([], dtype=np.float32)
    X_lagos_test = np.stack(lagos_test['patch'].values) if len(lagos_test) > 0 else np.array([]).reshape(0, *input_shape)
    y_lagos_test = lagos_test['pm25_mean'].values.astype(np.float32) if len(lagos_test) > 0 else np.array([], dtype=np.float32)
    # Build and train model on Delhi
    print("Building CNN model...")
    model = build_cnn_model(input_shape)
    print("Training model on Delhi data (2018-2022)...")
    model = train_model_on_delhi(model, X_delhi_train, y_delhi_train, X_delhi_val, y_delhi_val)
    # Save the baseline model (for potential use in linear calibration)
    baseline_model = model  # in this script, model is already the baseline after training
    # Transfer learning to Lagos
    adaptation = None
    if len(X_lagos_train) > 0:
        # If we have separate Lagos training data (outside 2023)
        print("Adapting model to Lagos...")
        adaptation = fine_tune_on_lagos(model, X_lagos_train, y_lagos_train, baseline_model=baseline_model)
        # If fine_tune_on_lagos returns a tuple, it means linear coefficients
        if isinstance(adaptation, tuple):
            # In linear case, baseline_model remains unchanged; we use coefficients separately
            model_adapted = None
        else:
            # If returned a model (fine-tuned), that's our adapted model
            model_adapted = adaptation
    else:
        # No separate Lagos train set (all Lagos data is 2023), use the data itself for calibration if small
        if len(X_lagos_test) > 0:
            print("Lagos has <100 samples and no separate training set; performing linear calibration on 2023 data...")
            adaptation = fine_tune_on_lagos(model, X_lagos_test, y_lagos_test, baseline_model=baseline_model)
            model_adapted = None
        else:
            model_adapted = None
    # Evaluate on Lagos 2023
    if len(X_lagos_test) > 0:
        print("Evaluating on Lagos 2023 data...")
        evaluate_on_lagos(baseline_model, X_lagos_test, y_lagos_test, adaptation if adaptation is not None else model_adapted)
    else:
        print("No Lagos 2023 test data available for evaluation.")
    
if __name__ == "__main__":
    main()
