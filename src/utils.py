import os

# utility function to dynamically load path for a model or file
def load_path(directory, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"../{directory}/{filename}")
    model_path = os.path.normpath(model_path)
    return model_path
