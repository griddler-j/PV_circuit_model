import a01_make_solar_cell as example1
import a02_make_PV_module as example2
import a03_make_tandem_cell as example3
from utilities import *

def test_1(pytest_mode=True):
    device = example1.main(display=False)
    print("test 1")
    run_record_or_test(device, this_file_prefix="a01",pytest_mode=pytest_mode)

def test_2(pytest_mode=True):
    device = example2.main(display=False)
    print("test 2")
    run_record_or_test(device, this_file_prefix="a02",pytest_mode=pytest_mode)

def test_3(pytest_mode=True):
    device = example3.main(display=False)
    print("test 3")
    run_record_or_test(device, this_file_prefix="a03",pytest_mode=pytest_mode)

if __name__ == "__main__":
    test_1(pytest_mode=False)
    test_2(pytest_mode=False)
    test_3(pytest_mode=False)