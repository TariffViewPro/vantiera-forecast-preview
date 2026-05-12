from preview_forecast import generate_preview_forecast

sample_history = """
120 132 118 141 150 147 155 160 158 162 170 168
"""

result = generate_preview_forecast(sample_history)

print(result)