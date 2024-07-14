import json
import random

class SearchSpaceParser:
    def __init__(self, model_file, hyperparameter_file):
        self.model_file = model_file
        self.hyperparameter_file = hyperparameter_file
        self.models = {}
        self.hyperparameters = {}
        self.search_space = {
            "models": [],
            "hyperparameters": {}
        }
        self.parse_search_space()

    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def parse_models(self):
        model_data = self.load_json(self.model_file)
        for model_name, model_info in model_data["Models"].items():
            if model_info["active"]:
                for variant in model_info["variants"]:
                    if variant["active"]:
                        self.search_space["models"].append(f"{model_name}_{variant['name']}")

    def parse_hyperparameters(self):
        hyperparameter_data = self.load_json(self.hyperparameter_file)
        for group_name, group_info in hyperparameter_data.items():
            self.search_space["hyperparameters"][group_name] = {}
            for param_name, param_info in group_info.items():
                if param_info["active"]:
                    self.search_space["hyperparameters"][group_name][param_name] = param_info["options"]

    def parse_search_space(self):
        self.parse_models()
        self.parse_hyperparameters()
        return self.search_space
    
    def sample_search_space(self):
        """
        Randomly sample the search space to get a model and hyperparameters configuration.
        """
        model = random.choice(self.search_space["models"])
        hyperparameters = {}
        for group_name, group_info in self.search_space["hyperparameters"].items():
            hyperparameters[group_name] = {}
            for param_name, param_info in group_info.items():
                hyperparameters[group_name][param_name] = random.choice(param_info)
        return model, hyperparameters
    
    def get_total_search_space(self):
        total_search_space = 1
        for group_name, group_info in self.search_space["hyperparameters"].items():
            total_search_space *= len(group_info)
        total_search_space *= len(self.search_space["models"])
        return total_search_space

# Example usage
if __name__ == "__main__":
    model_file = "search_space/models_ss.json"
    hyperparameter_file = "search_space/hyperparameters_ss.json"
    search_space_parser = SearchSpaceParser(model_file, hyperparameter_file)
    search_space = search_space_parser.parse_search_space()
    print(search_space)
    total_search_space = search_space_parser.get_total_search_space()
    print(f"Total search space: {total_search_space}")
    print("Sampled search space:")
    model, hyperparameters = search_space_parser.sample_search_space()
    print(f"Model: {model}")
    print("Hyperparameters:")
    print(json.dumps(hyperparameters, indent=4))
