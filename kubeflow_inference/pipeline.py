from kfp.components import OutputPath, InputPath
from run_inference import run_blocked_inference
from kfp import dsl, compiler 
from kfp.components import create_component_from_func
from blocking import blocking_func
from types import SimpleNamespace

def run_blocking_component(
    table_reference_csv: InputPath('CSV'),
    table_source_csv: InputPath('CSV'),
    output_pairs_csv: OutputPath('CSV'),
    output_ditto_txt: OutputPath(str),
    table_reference_txt: OutputPath(str),
    table_source_txt: OutputPath(str),
    table_reference_vec: OutputPath(str),
    table_source_vec: OutputPath(str),
    model_name_blocking: str,
    threshold_blocking: float,
    top_k_blocking: int,
    batch_size_blocking: int
):
    hp = SimpleNamespace(
        table_reference_csv=table_reference_csv,
        table_source_csv=table_source_csv,
        output_pairs_csv=output_pairs_csv,
        output_ditto_txt=output_ditto_txt,
        table_reference_txt=table_reference_txt,
        table_source_txt=table_source_txt,
        table_reference_vec=table_reference_vec,
        table_source_vec=table_source_vec,
        model_name_blocking=model_name_blocking,
        threshold_blocking=threshold_blocking,
        top_k_blocking=top_k_blocking,
        batch_size_blocking=batch_size_blocking
    )
    blocking_func(hp)
    
def run_blocked_inference_component(
    model_path: str,
    blocked_pairs_csv: InputPath('CSV'),
    reference_table_csv: InputPath('CSV'),
    source_table_csv: InputPath('CSV'),
    output_csv: OutputPath('CSV'),
    lm: str,
    max_len: int
):
    run_blocked_inference(
        model_path=model_path,
        blocked_pairs_csv=blocked_pairs_csv,
        reference_table_csv=reference_table_csv,
        source_table_csv=source_table_csv,
        output_csv=output_csv,
        lm=lm,
        max_len=max_len
    )

run_blocking_op = create_component_from_func(
    func=run_blocking_component,
    base_image='172.17.232.16:9001/ditto:latest'
)

run_inference_op = create_component_from_func(
    func=run_blocked_inference_component,
    base_image='172.17.232.16:9001/ditto:latest'
)

@dsl.pipeline(
    name='Blocked Inference Pipeline',
    description='Pipeline to run blocking then inference on candidate pairs'
)
def blocked_inference_pipeline(
    model_path,
    reference_table_csv,
    source_table_csv,
    lm,
    max_len,
    model_name_blocking,
    threshold_blocking,
    top_k_blocking,
    batch_size_blocking
):
    # Step 1: Run blocking
    blocking_task = run_blocking_op(
        table_reference_csv=reference_table_csv,
        table_source_csv=source_table_csv,
        model_name_blocking=model_name_blocking,
        threshold_blocking=threshold_blocking,
        top_k_blocking=top_k_blocking,
        batch_size_blocking=batch_size_blocking
    )

    # Specify hardware for blocking task
    blocking_task.set_gpu_limit(1)           # Request 1 GPU
    blocking_task.set_cpu_limit('2')         # Request 2 CPU cores
    blocking_task.set_memory_limit('8Gi')    # Request 8 GB memory
    blocking_task.set_display_name("Blocking Step")

    # Step 2: Run inference using output of blocking
    inference_task = run_inference_op(
        model_path=model_path,
        blocked_pairs_csv=blocking_task.outputs['output_pairs_csv'],
        reference_table_csv=reference_table_csv,
        source_table_csv=source_table_csv,
        lm=lm,
        max_len=max_len
    ).after(blocking_task)

    # Specify hardware for inference task
    inference_task.set_gpu_limit(1)           # Request 1 GPU
    inference_task.set_cpu_limit('4')         # Request 4 CPU cores
    inference_task.set_memory_limit('16Gi')   # Request 16 GB memory
    inference_task.set_display_name("Inference Step")
    
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=blocked_inference_pipeline,
        package_path='blocked_inference_pipeline.yaml'
    )