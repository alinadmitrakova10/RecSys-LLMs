# RecSys-LLMs
Repository for Recommender Systems course in HSE : Code examples, assignments, and capstone project templates using JavaScript, Hugging Face LLM API, and GitHub Pages. From random lunch recommender to personalized AI systems.


## My prompt
The project consists of the original prompt and the application code in `index.html`. The application is a Random Lunch Generator: when the user clicks the button, it randomly selects and displays a lunch option from the existing menu.

I manually opened the provided Random Lunch Generator in a browser and tested the lunch options. I observed that the icons for **Ramen**, **Pasta**, and **Soup** did not appear, while the lunch names were still displayed.

My hypothesis is that the application references icon class names that are unavailable in the connected Font Awesome Free 6.4.0 library. Please verify this hypothesis and correct the existing starter code.

Requirements

1. Work only with the provided `index.html`. Do not recreate or redesign the application.
2. Inspect all entries in the existing `lunchMenu` array, not only the three items I identified.
3. Determine which icon classes render and which do not render with Font Awesome Free 6.4.0.
4. Check the failing classes against official Font Awesome documentation and explain the exact mismatch.
5. Replace only unavailable icon classes with semantically appropriate icons that are included in the free library.
6. Preserve the current HTML structure, CSS design, lunch names, random-selection logic, animation, and page behavior.
7. Verify the corrected class names for all lunch entries.
8. Report the baseline and corrected results as counts and percentages.
9. Show the exact code changes and distinguish my original observation from the AI-assisted verification.

Do not claim that the issue is fixed unless every menu entry has been checked. Do not create a new implementation from scratch.
