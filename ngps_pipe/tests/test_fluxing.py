import os

from pkg_resources import resource_filename

from pypeit import sensfunc

from ngps_pipe import fluxing

def test_archived_sensfunc_exist():
    for sensfun in fluxing.archived_sensfuncs:
        assert os.path.isfile(resource_filename("ngps_pipe",
            f"data/sens_{sensfun}.fits")), \
            f"Archived sensitivity function sens_{sensfun}.fits could not be found."

def test_archived_sensfunc_read():
    for sensfun in fluxing.archived_sensfuncs:
        sens_path = resource_filename("ngps_pipe", f"data/sens_{sensfun}.fits")
        sensobj = sensfunc.SensFunc.from_file(sens_path)

        assert sensobj.wave.shape[1] == 1, ("Archived sensitivity function "
            f"sens_{sensfun}.fits must have shape (nspec, 1)")
        assert len(sensobj.wave.shape) == 2, ("Archived sensitivity function "
            f"sens_{sensfun}.fits must have shape (nspec, 1)")

        assert sensobj.zeropoint.shape == sensobj.wave.shape, (f"Archived "
            f"sensitivity function sens_{sensfun}.fits must have zeropoint and "
            "wavelength of same shape")
