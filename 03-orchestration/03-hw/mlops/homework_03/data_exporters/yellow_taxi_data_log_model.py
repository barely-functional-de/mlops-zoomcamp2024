import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    lr, dv, intercept_, X_train, y_train = data
    print(intercept_)
    with mlflow.start_run():
        lr.fit(X_train, y_train)

        mlflow.sklearn.log_model(lr, "linear_regression_model")
        # Save the DictVectorizer as an artifact
        # dv_path = "dict_vectorizer.pkl"
        # joblib.dump(dv, dv_path)
        # mlflow.log_artifact(dv_path)



