python retrain.py ^
--bottleneck_dir bottleneck ^
--how_many_training_steps 200 ^
--model_dir inception_model ^
--output_graph new_model/output_graph.pb ^
--output_labels output_labels.txt ^
--image_dir train/ ^
--summaries_dir logs/
pause

