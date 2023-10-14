# Create a Python script to inspect model.py
with open('/Users/jashparekh/Documents/GitHub/dermalize/Backend/model.py', 'r') as f:
    content = f.read()

# Execute the content to get class names
exec(content)

# Print the names of classes defined in model.py
import inspect
model_classes = [name for name, obj in inspect.getmembers(content) if inspect.isclass(obj)]
print(model_classes)
