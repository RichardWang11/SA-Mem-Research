import os

def get_answer_dir(answer_path, dataset_name):
    return os.path.join(answer_path, dataset_name)

def get_result_file(answer_dir, config_filename):
    return os.path.join(answer_dir, f"{config_filename}_results.json")

def get_cache_folder(cache_path, data_file, date, piece_name):
    return os.path.join(cache_path, data_file, str(date), str(piece_name))

def get_query_time_file(answer_dir, config_filename):
    return os.path.join(answer_dir, f"{config_filename}_query_times.txt")

def get_config_filename(data_file, desc, config_hash):
    return f"{data_file}_{desc}_{config_hash}"