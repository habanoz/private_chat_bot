import argparse


def main():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to read data from')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    parser.add_argument('--index_dir', type=str, default='index', help='Directory to save index')
    parser.add_argument('--retrieval_mode', type=str, default='hybrid', choices=['hybrid', 'dense'],
                        help='Retrieval mode')
    parser.add_argument('--st_model_name', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='sentence-transformer embedding model')