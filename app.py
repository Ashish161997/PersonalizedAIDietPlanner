import os 
import sys 
from crewai import Agent, Task, Crew, Process, LLM
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

api_key = os.getenv("api_key")
my_llm = LLM(
              model='gemini/gemini-1.5-flash',
              api_key=api_key
            )





def generate_diet_plan(age, gender, weight, height, activity_level, goal, diet_pref, disease):
    profile_text = f"Age: {age}\nGender: {gender}\nWeight: {weight}\nHeight: {height}\nActivity Level: {activity_level}\nGoal: {goal}\nDiet Preference: {diet_pref}\nHealth Condition/Disease: {disease}"

        # Agents
    nutritionist_agent = Agent(
        role="Certified Nutritionist",
        goal="Determine the user's ideal daily calorie and macronutrient requirements",
        backstory=( 
            "You are a certified nutritionist with 10 years of experience in personalized diet planning. "
            "You specialize in calculating daily energy expenditure and optimal macronutrient ratios "
            "based on the user's age, gender, height, weight, activity level, and fitness goals such as fat loss, muscle gain, or maintenance."
            "Use standard formulas like Mifflin-St Jeor and apply logical reasoning to suggest daily calories, protein, carbs, and fat requirements."
        ),
        llm=my_llm,
        allow_delegation=False,
        verbose=False
    )

    mealplanner_agent = Agent(
        role="Smart Meal Planner",
        goal="Generate a 7-day meal plan aligned with the user's nutrition goals and dietary preferences",
        backstory=( 
            "You are an expert meal planning assistant who crafts weekly diet plans based on specific calorie and macronutrient targets. "
            "You tailor meals to suit dietary restrictions like vegetarian, vegan, gluten-free, and keto. "
            "You ensure variety, cultural inclusivity, and nutritional balance in each meal."
            "Use food items that are commonly available and provide breakfast, lunch, dinner, and optional snacks."
        ),
        llm=my_llm,
        allow_delegation=False,
        verbose=False
    )

    shoppinglist_agent = Agent(
        role="Grocery List Assistant",
        goal="Convert meal plans into an organized weekly shopping list",
        backstory=( 
            "You are a detail-oriented assistant who compiles ingredient lists from daily meals into a user-friendly shopping list. "
            "You group items by category (vegetables, fruits, grains, dairy, etc.) and include quantities needed for a 7-day meal plan. "
            "You ensure the list is optimized to reduce food waste and avoid unnecessary items."
        ),
        llm=my_llm,
        allow_delegation=False,
        verbose=False
    )

    # Medical Check Agent (new)
    medical_agent = Agent(
        role="Diet Correction Assistant",
        goal="Modify meal plans to ensure they are medically appropriate for the user's condition",
        backstory=( 
            "You are a clinical diet assistant who adjusts meal plans for users with health conditions like diabetes or hypertension. "
            "You do not write medical assessments. Your only job is to review the given 7-day meal plan and remove or replace meals or ingredients "
            "that may worsen the user's health condition. You strictly follow common dietary guidelines such as low sugar for diabetes or low sodium for hypertension. "
            "Return only the modified 7-day meal plan with no extra explanations."
        ),
        llm=my_llm,
        allow_delegation=False,
        verbose=False
    )


    # Tasks
    nutrition_task = Task(
        description=( 
            f"User Profile:\n{profile_text}\n\n"
            "1. Analyze the user's age, gender, weight, height, activity level, and dietary goals.\n"
            "2. Estimate their Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE).\n"
            "3. Determine optimal macronutrient distribution based on their goal (fat loss, muscle gain, maintenance).\n"
        ),
        expected_output="A daily nutrition summary including BMR, TDEE, total daily calories, and a breakdown of protein, carbs, and fats.",
        agent=nutritionist_agent
    )

    meal_plan_task = Task(
        description=( 
            f"User Profile:\n{profile_text}\n\n"
            "1. Use the nutrition summary (calories and macronutrients) from the nutritionist.\n"
            "2. Generate a 7-day meal plan with breakfast, lunch, dinner, and snacks.\n"
            "3. Ensure meals align with nutritional goals and dietary preferences.\n"
        ),
        expected_output="A complete and structured 7-day meal plan with meals, ingredients, and calories/macros.",
        agent=mealplanner_agent
    )

    shopping_list_task = Task(
        description=(
            "1. Parse the meal plan and extract unique ingredients.\n"
            "2. Group ingredients by category (vegetables, fruits, grains, dairy).\n"
            "3. Ensure the list is optimized for food waste and simplicity."
        ),
        expected_output="An organized grocery list with quantities for each ingredient.",
        agent=shoppinglist_agent
    )

    medical_check_task = Task(
        description=(
            f"User Profile:\n{profile_text}\n\n"
            "1. Receive the full 7-day meal plan generated earlier.\n"
            "2. Identify any meals or ingredients that are not suitable for the user's condition (e.g., high sugar for diabetes, high sodium for hypertension).\n"
            "3. Modify those meals or ingredients with medically suitable alternatives.\n"
            "4. Do not provide commentary or assessments — only return the revised meal plan."
        ),
        expected_output=(
            "A revised 7-day meal plan with all unsafe items removed or replaced based on the user's health condition."
        ),
        agent=medical_agent
    )


    # Update task descriptions directly
    # nutrition_task.description = nutrition_task.description.replace("{profile_text}", profile_text)
    # meal_plan_task.description = meal_plan_task.description.replace("{profile_text}", profile_text)
    # medical_check_task.description = medical_check_task.description.replace("{profile_text}", profile_text)

    # Run the Crew
    crew = Crew(
        agents=[nutritionist_agent, mealplanner_agent, shoppinglist_agent, medical_agent],
        tasks=[nutrition_task, meal_plan_task, shopping_list_task, medical_check_task],
        verbose=True
    )
    result=crew.kickoff()
    return result.raw

from markdown import markdown
with gr.Blocks() as demo:
    gr.Markdown("## 🥗 Personalized AI Diet Planner")
    
    with gr.Row():
        age = gr.Number(label="Age", value=30)
        gender = gr.Dropdown(["male", "female"], label="Gender", value="male")
        weight = gr.Number(label="Weight (kg)", value=70)
        height = gr.Number(label="Height (cm)", value=175)

    with gr.Row():
        activity_level = gr.Dropdown(
            ["sedentary", "lightly active", "moderately active", "very active"],
            label="Activity Level", value="moderately active"
        )
        goal = gr.Dropdown(
            ["fat loss", "muscle gain", "maintenance"],
            label="Goal", value="fat loss"
        )
        diet_pref = gr.Dropdown(
            ["no restriction", "vegetarian", "vegan", "gluten-free", "keto"],
            label="Diet Preference", value="vegetarian"
        )
        disease = gr.Textbox(label="Health Condition/Disease", placeholder="e.g., Diabetes, Hypertension")

    generate = gr.Button("Generate My Plan")
    # output = gr.Textbox(label="Your Personalized Diet Plan", lines=30)
    output = gr.HTML(label="Your Diet Plan") 
    def format_output(plan_text):
        # Convert Markdown to HTML
        html_content = markdown(plan_text)
        
        # Add custom styling
        styled_html = f"""
        <div style='
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        '>
            {html_content}
        </div>
        """
        return styled_html

    generate.click(
        fn=lambda *args: format_output(generate_diet_plan(*args)),
        inputs=[age, gender, weight, height, activity_level, goal, diet_pref, disease],
        outputs=output
    )

demo.launch()
