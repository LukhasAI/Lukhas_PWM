import pytest
import os

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = os.getcwd()
    pytest.main(["-v", "--maxfail=10", "tests/active/"])
