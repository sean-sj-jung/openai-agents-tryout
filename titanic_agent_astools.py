from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import asyncio
from openai.types.responses import ResponseTextDeltaEvent


@function_tool
def save_file(content: str, filename: str):
    """
    Saves the given content to a file.
    Parameters:
        content (str): The text content to save.
        filename (str): The name of the file to save the content to.
    """
    try:
        with open(filename, "w") as file:
            file.write(content)
        print(f"Content saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving content: {e}")


@function_tool
def execute_code(code: str) -> str:
    """Executes the provided Python code and returns the output."""
    try:
        # Use 'exec' to execute the code within a local namespace
        local_namespace = {}
        exec(code, {}, local_namespace)
        # Capture the output from the local namespace
        output = local_namespace.get('output', 'No output variable set.')
        return str(output)
    except Exception as e:
        return f"Error executing code: {e}"


@function_tool
def execute_terminal(command: str) -> str:
    """Executes terminal commands and returns the output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout or result.stderr
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"

ml_modeler = Agent(name="MachineLearningModeler", 
                instructions="You are a machine learning modeler. You are given a dataset and you need to build a model to predict the target variable. \
                You need to use the dataset to build a model that is able to predict the target variable with the highest accuracy. Begin with XGBoost and build from there.\
                Try at least five variations to improve the accuracy of the model without overfitting.",
                handoff_description="Agent for writing python code and building machine learning model",
                tools=[save_file, execute_code, execute_terminal])

async def main():
    result = Runner.run_streamed(ml_modeler, input="Using the titanic_train.csv dataset, build a xgboost model to predict the target variable, Survived. \
                                 Save the model to a file called titanic_model.pkl. Make predictions on the titanic_test.csv dataset and save the predictions to a file called titanic_predictions.csv."
                                 )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())