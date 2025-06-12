import h2o
h2o.init()

def load_occ_model(model_occ_path='models/failure_occurrence_model.mojo'):
    """Load H2O occurence model from the given path."""
    return h2o.import_mojo(model_occ_path)

def load_fail_model(model_fail_path='models/failure_type_model.mojo'):
    """Load H2O failure type model from the given path."""
    return h2o.import_mojo(model_fail_path)

