from shiny import run_app
from barbe.utils.visualizer_utils import *


def test_visualizer():
    run_app("visualizer", app_dir="../")


def test_open_data():
    data, names, types, ranges = open_input_file('../dataset/glass.data')
    print(data)
    print(names)
    print(types)
    print(ranges)
    print(list(ranges[-1].astype(str)))


def test_bad_data():
    data, names, types, ranges = open_input_file('../dataset/nope')
    print(data)
    print(names)
    print(types)
    print(ranges)
