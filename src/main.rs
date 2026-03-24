use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use chrono::{Utc, DateTime};

// Enum to represent different types of ML tasks
enum MlTask {
    Training,
    Inference,
    Preprocessing,
    Evaluation,
}

impl MlTask {
    fn to_string(&self) -> &str {
        match self {
            MlTask::Training => "Training",
            MlTask::Inference => "Inference",
            MlTask::Preprocessing => "Preprocessing",
            MlTask::Evaluation => "Evaluation",
        }
    }
}

// Struct to represent a single step in an ML pipeline
struct PipelineStep {
    name: String,
    task_type: MlTask,
    command: String,
    dependencies: Vec<String>,
}

impl PipelineStep {
    fn new(name: &str, task_type: MlTask, command: &str, dependencies: Vec<&str>) -> Self {
        PipelineStep {
            name: name.to_string(),
            task_type,
            command: command.to_string(),
            dependencies: dependencies.iter().map(|&s| s.to_string()).collect(),
        }
    }

    fn execute(&self) -> Result<(), String> {
        println!("[{}][{}] Executing step: {}", Utc::now().format("%Y-%m-%d %H:%M:%S"), self.task_type.to_string(), self.name);
        println!("Command: {}", self.command);

        let output = if cfg!(target_os = "windows") {
            Command::new("cmd")
                    .args(&["/C", &self.command])
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .output()
        } else {
            Command::new("sh")
                    .arg("-c")
                    .arg(&self.command)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .output()
        };

        match output {
            Ok(output) => {
                if output.status.success() {
                    println!("[{}][{}] Step '{}' completed successfully.", Utc::now().format("%Y-%m-%d %H:%M:%S"), self.task_type.to_string(), self.name);
                    Ok(())
                } else {
                    let error_msg = format!("[{}][{}] Step '{}' failed with exit code: {:?}", Utc::now().format("%Y-%m-%d %H:%M:%S"), self.task_type.to_string(), self.name, output.status.code());
                    eprintln!("{}", error_msg);
                    Err(error_msg)
                }
            },
            Err(e) => {
                let error_msg = format!("[{}][{}] Failed to execute command for step '{}': {}", Utc::now().format("%Y-%m-%d %H:%M:%S"), self.task_type.to_string(), self.name, e);
                eprintln!("{}", error_msg);
                Err(error_msg)
            }
        }
    }
}

// Struct to represent an ML pipeline
struct MlPipeline {
    name: String,
    steps: HashMap<String, PipelineStep>,
    execution_order: Vec<String>,
}

impl MlPipeline {
    fn new(name: &str) -> Self {
        MlPipeline {
            name: name.to_string(),
            steps: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    fn add_step(&mut self, step: PipelineStep) {
        self.steps.insert(step.name.clone(), step);
    }

    fn topological_sort(&mut self) -> Result<(), String> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for step in self.steps.values() {
            in_degree.entry(step.name.clone()).or_insert(0);
            graph.entry(step.name.clone()).or_insert_with(Vec::new);
            for dep in &step.dependencies {
                *in_degree.entry(dep.clone()).or_insert(0) += 1;
                graph.entry(dep.clone()).or_insert_with(Vec::new).push(step.name.clone());
            }
        }

        let mut queue: Vec<String> = in_degree.iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let mut sorted_list: Vec<String> = Vec::new();
        while let Some(node) = queue.pop() {
            sorted_list.push(node.clone());
            if let Some(neighbors) = graph.get(&node) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }
        }

        if sorted_list.len() != self.steps.len() {
            Err("Circular dependency detected in pipeline.".to_string())
        } else {
            self.execution_order = sorted_list;
            Ok(())
        }
    }

    fn run(&mut self) -> Result<(), String> {
        println!("\n[{}][Orchestrator] Starting ML Pipeline: {}", Utc::now().format("%Y-%m-%d %H:%M:%S"), self.name);

        self.topological_sort()?;

        for step_name in &self.execution_order {
            let step = self.steps.get(step_name).ok_or(format!("Step '{}' not found.", step_name))?;
            step.execute()?;
        }

        println!("[{}][Orchestrator] ML Pipeline '{}' completed successfully.", Utc::now().format("%Y-%m-%d %H:%M:%S"), self.name);
        Ok(())
    }
}

fn main() -> Result<(), String> {
    // Example Pipeline Definition
    let mut pipeline = MlPipeline::new("Image Classification Workflow");

    let step1 = PipelineStep::new(
        "Data Preprocessing",
        MlTask::Preprocessing,
        "python preprocess.py --input_dir data/raw --output_dir data/processed",
        vec![],
    );
    let step2 = PipelineStep::new(
        "Model Training",
        MlTask::Training,
        "python train.py --config configs/resnet50.yaml",
        vec!["Data Preprocessing"],
    );
    let step3 = PipelineStep::new(
        "Model Evaluation",
        MlTask::Evaluation,
        "python evaluate.py --model_path models/resnet50.pth",
        vec!["Model Training"],
    );
    let step4 = PipelineStep::new(
        "Model Deployment",
        MlTask::Inference,
        "docker build -t image-classifier . && docker push myregistry/image-classifier",
        vec!["Model Evaluation"],
    );

    pipeline.add_step(step1);
    pipeline.add_step(step2);
    pipeline.add_step(step3);
    pipeline.add_step(step4);

    // Run the pipeline
    pipeline.run()
}
