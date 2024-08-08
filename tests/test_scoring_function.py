import os
import shutil
import tempfile

from acegen.scoring_functions import (
    Task,
    register_custom_scoring_function,
    custom_scoring_functions,
)


def test_scoring_functions_utils():
    temp_dir = tempfile.mkdtemp()
    register_custom_scoring_function("QED", "acegen.scoring_functions.chemistry.QED")
    task = Task(
        name="QED2",
        scoring_function=custom_scoring_functions["QED"],
        budget=4,
        output_dir=temp_dir,
    )
    assert not task.finished
    counter = 0
    for i in range(4):
        score = task(["CC1=CC=CC=C1"])
        assert len(score) == 1
        counter += 1
    assert counter == 4
    assert task.finished
    assert os.path.isfile(f"{temp_dir}/compounds.csv")
    shutil.rmtree(temp_dir)