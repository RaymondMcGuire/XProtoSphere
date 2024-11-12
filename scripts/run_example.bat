@echo off

cd ..

set input_model_name="bunny"
set input_model_scale=0.6
set distribution_type="A"
set protosphere_max_iter=10
set enable_dem_relax=1
set overlap_ratio=0.6
set dsr_max_iter=1000
set enable_dist_constain=0

call .\build\bin\Release\xprotosphere.exe %input_model_name% %input_model_scale% %distribution_type% %protosphere_max_iter% %enable_dem_relax% %overlap_ratio% %dsr_max_iter% %enable_dist_constain%



pause