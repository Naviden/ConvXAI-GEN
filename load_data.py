import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## CARICAMENTO DATASET [IN FORMATO SKLEARN]

def load_dataset_as_sklearn_format(csv_path):
    """
    Load a dataset from CSV and convert it to sklearn format

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        dict: A dictionary containing train/test splits and metadata
    """
    # Determine the target column based on the dataset
    if "Iris" in csv_path or "iris" in csv_path.lower():
        target_column = "Species"
        delimiter = ","
    elif "winequality" in csv_path or "wine" in csv_path.lower():
        target_column = "color"
        delimiter = ";"
    else:
        raise ValueError(f"Unknown dataset for path: {csv_path}")

    # Read the CSV
    df = pd.read_csv(csv_path, delimiter=delimiter)

    # Check if target column exists
    if target_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Rimuove la colonna "Id" se presente
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        'feature_names': X.columns.tolist(),
        'target_names': label_encoder.classes_,
        'label_encoder': label_encoder
    }
