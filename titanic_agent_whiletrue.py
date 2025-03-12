from agents import Agent, Runner, function_tool, ItemHelpers, TResponseInputItem
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
                instructions="You are a machine learning modeler. You are given a dataset and tool to execute python code. Your task is to build a model to predict the target variable. \
                You need to use the dataset to build a model that is able to predict the target variable with the highest accuracy. Begin with XGBoost and build from there.\
                Try at least five variations to improve the accuracy of the model without overfitting.",
                # handoff_description="Agent for writing python code and building machine learning model",
                tools=[save_file, execute_code])


terminal_agent = Agent(name="TerminalAgent",
              instructions="Your task is to install missing python packages. You have access to terminal. Only provide pip install commands. Any other commands are strictly forbidden.",
              handoff_description="Agent for running installing python library by accessing terminal and running pip install commands",
              tools=[execute_terminal])


triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request. \
        Terminate the conversation when the train accuract is over 0.95",
    handoffs=[ml_modeler],
)


async def main():
    input0 = "Using the titanic_train.csv dataset, build a xgboost model to predict the target variable, Survived. Save the model to a file called titanic_model.pkl. Make predictions on the titanic_test.csv dataset and save the predictions to a file called titanic_predictions.csv."
    input_items: list[TResponseInputItem] = [{"content": input0, "role": "user"}]
    while True:
        result = Runner.run_streamed(ml_modeler, input=input_items)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

        input_items = result.to_input_list()
        latest_item = ItemHelpers.text_message_outputs(result.new_items)
        input_items.append({"content": latest_item, "role": "user"})


if __name__ == "__main__":
    asyncio.run(main())