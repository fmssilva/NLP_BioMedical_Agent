					
	Before implementing anything, read the project and specific phase guide:				
		C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\_PROJ_PLAN.md			
		C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\_PROJ_STRUCTURE_PLAN.md			
		C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\_PHASE_2_PLAN.md			
		and also the project guides: 			
			C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\proj_guide_NLPS_-_TREC_2025_BioGen.md		
		and all relevant existing files. Understand the full project structure before touching any code. Think as a software architect first, not a coding agent.			
					
	Code Quality				
		Simple, clean, well-organized. No over-engineering.			
		Use what Python/PyTorch/sklearn already give you, or NLP things like OpenSearch, ranx, transformers — don't reimplement what libraries handle well.			
		Small files with single, clear responsibilities. Name everything so the structure is self-documenting.			
		Domain-centered structure (not tests/, plots/ folders — keep related things together).			
		Only implement what is currently needed. No "might be useful later" functions.			
		Before you implement anything, check the files in the folder of the labs of the class. They are the examples of code we should follow as close as possible:			
			"C:\Users\franc\Desktop\NLP_Biomedical_Agent\references
» see the file with an index of the labs code that we should follow: C:\Users\franc\Desktop\NLP_Biomedical_Agent\references\README.md"		
					
	Comments & Logs				
		Short comment above each function (as concise as possible).			
		Inline comments during the code itself to explain the code. This is a teaching project so i want to have comments to explain all important details of each block of code we do so a reader can understand it easy, BUT WRITTEN AS CONCISE AS POSSIBLE, AND WRITTEN AS CASUAL AND NATURAL LANGUAGE AS POSSIBLE, LIKE IF IT WAS ONE DEV TALKING TO ANOTHER DEV. SO CONCISE BULLET POINTS, NO FORMAL PROSE. NO EMOJIS ANYWHERE!!! AND NO NON-ASCII SYMBOLS (example ✓ → [ok], → → -> or >>, etc)			
		Logs: only what's needed to pinpoint errors. Single-line, concise, no emojis (terminal encoding issues).			
					
	Notebook first flow				
		Start by making or correcting or improving the notebook cells to implement these steps and tasks. 			
		then the src code necessary, then compreensive local unit and integration tests to confirm everything is working correctly, and using some small data sample, and faster hyper params, etc... then a final run on the notebook cells of these steps/tasks we are doing now and others that might have been afected, and confirm we have the corret outputs, in terms of formating of output without duplictions and in a clean presentation, and then also observing the concrete values to confirm our logic is actually correct and to help us do the necessary analysis. 			
		in the markdown structure and markdown cells and code cells, and analysis cells etc... write all the important details in a concise manner. 			
		lets have the notebook like some sort of learning tool where a user start reading and understands evrything we are doing, including some simple intituiton or theory if necessary. BUT WRITTEN AS CONCISE AS POSSIBLE, AND WRITTEN AS CASUAL AND NATURAL LANGUAGE AS POSSIBLE, LIKE IF IT WAS ONE DEV TALKING TO ANOTHER DEV. SO CONCISE BULLET POINTS, NO FORMAL PROSE. NO EMOJIS ANYWHERE!!! AND NO NON-ASCII SYMBOLS (example ✓ → [ok], → → -> or >>, etc); (we can yes have formulas well formated  if needed for better reading. 			
					
	File & Folder Structure				
		Prefer this pattern — everything by domain (this is an example for older project just to show the idea):			
		src/			
			datasets/		
				dataset.py       # loading + transforms	
				eda.py           # EDA functions	
				eda_plots.py     # EDA visualizations	
				__xxx_test.py       # if some file needs more tests than just a few in a __main__ section at the end of the file, so we implement more compreensive tests in its own file 	
		No utils.py dumping ground with 50 mixed functions. If a file starts doing too many unrelated things, split it.			
					
	Testing During Development				
		Along the code add some asserts to confirm the main possible sources of errors. example assert (some_arg is not Null), f"{some_arg} is Null"			
		Then write local tests for each function and file. 			
			For that, if we need only a small number of tests, we can put them directly in the file, in the  if __name__ == "__main__" block. If it is good to implement a good number of tests so we can create a separate file for that __xxx_test.py		
				"and if we put some tests in the __main__ so lets put a clear separation in the file example: 
###############################################
##                               LOCAL TESTS                               ##
###############################################
and we write after this separation all the helper functions and constants that we have that are used only for the tests and then the __main__ section... (and if in its own file so lets put these helper functions and posible constants used only for the tests in that test file and not mixed with the real code "	
			Good tests to include (adapt this to the kind of project we are building, example pure DL vs NLP, etc.):		
				Connection tests:   if we are using some external thing like OpenSearch (reachable, index exists, doc count matches expected...)	
				Data tests: shapes correct, labels encoded right, no NaNs, class distribution matches CSV, correct splits, correct qrels from groudtruth... 	
				Transform tests: output tensor shape, value range [0,1] or normalized	
				Model tests: forward pass with dummy input gives expected output shape	
				Training tests: loss decreases after 1-2 steps on a tiny batch (sanity check)	
				Submission tests: output CSV has correct number of rows, valid class names, correct format	
				Retrieval tests:    each strategy returns results, no empty hits, scores decrease monotonically down the ranked list	
				Evaluation tests:   ranx run file format valid, graded relevance passed correctly, P@10 + R@100 + nDCG all computed without errors	
				Generation tests:   answer ≤250 words, ≤3 PMIDs per sentence, all cited PMIDs exist in corpus	
				Agent tests:        planner returns ≥1 sub-topic, ReAct loop terminates, final report has citations	
				Plots tests: confirm that all the plots render correctly, with correct values and correct visuals and desgin	
			How to run the tests:		
				First run the tests with a small sample of data, just to confirm all the code is correct, no errors, all the plots render, all the params match, etc. that we don't have 	
				When we have 100% sure all tests pass in terms of code, so we run them with the minimum amount of data necessary to actually tests the things we want. Example some 2 or 3 epochs to confirm our training works well with only some 5-10 samples, etc. Or if needed with the full data example if we are actually analysing data statistics, etc. 	
					
	Before Implementing Any Task				
		Read all relevant existing files — avoid duplication.			
		Think the 3 best options at architecture level (where/how it fits in the project).			
		Think the 3 best options at implementation level (how to write the code locally).			
		Choose the best one — clean, simple, maintainable. No shortcuts, no over-engineering.			
					
	Workflow				
		"1. implement notebook and respective necessary src code (ONLY THE NECESSARY SRC CODE TO HELP RUN THE NOTEBOOK CELLS. NOTHING MORE).
2. implement compreensive tests to confirm that all the code in the src folder AND THE CODE IN THE NOTEBOOK CELLS ARE ALL CORRECT AND WORKING WELL AND PRODUCING THE CORRECT OUTPUTS... (we need to test also the code in the notebook cells, if needed we can duplicate that code of the notebook cells in some src __xxx_test.py file and confirm that everything is working well, before we run the notebook). 
3. run these local unit and integration tests, on CPU, with small data... (to be fast) → fix problems
4. run notebook cells, on CPU, with small data... (to be fast) → fix problems
5. → only after I'll run the notebook on Colab / GPU"			
					
	Terminal				
		PowerShell — use ; not && for chaining commands.			
		avoid to run commands with things like 2>&1 and | and filters... prefer to have the logs go normally to the terminal directly so i can also see better what is happening 			
		i have alrady a environment set up with anaconda: lets use "cnn (3.10.19)", so don't create new environemnts 			
		"When Editing Jupyter Notebooks in VS Code
» The Only Tool You Need `edit_notebook_file` — edits notebooks live in VS Code. Changes appear instantly.  Never manually edit the `.ipynb` JSON on disk. Step 0 — Always get the notebook state first - Call `copilot_getNotebookSummary` before any edit to get cell IDs, types, and order. You need the cell ID (e.g. `#VSC-3f7cadd8`) to target specific cells. Operations:
| Action              | `editType` | `cellId`                       |
| ------------------- | ---------- | ------------------------------ |
| Edit existing cell  | `edit`     | `#VSC-<id>`                    |
| Insert at top       | `insert`   | `TOP`                          |
| Insert at bottom    | `insert`   | `BOTTOM`                       |
| Insert after a cell | `insert`   | `#VSC-<id>` (inserts after it) |
| Delete a cell       | `delete`   | `#VSC-<id>`                    |
**Parameters always needed:** `filePath` (absolute), `editType`, `cellId`, `language` (`python` or `markdown`), `newCode` (full cell content — not a diff).
Run & Read cells:
- **Run a cell:** `run_notebook_cell` → needs `filePath` + `cellId`
  - Call `configure_notebook` **once per session** before first run
- **Read output:** `read_notebook_cell_output` → needs `filePath` + `cellId`
Key Rules
- Re-run `copilot_getNotebookSummary` after inserts/deletes (IDs/positions change)
- `newCode` replaces the **entire** cell — include everything you want
- Markdown cells **cannot** be executed
- Always use **absolute file paths**"			
		Sometimes these edits doesn't work. So before you start working, just find some code cell, and add a simple print("Hello") and run the cell and see if your edit took effect. because sometimes your edits aren't working don't know why. if you run the cell and you don't see the "Hello" log, so lets try to edit the .ipynb JSON on disk directly via Python script. Those changes use to take effect imediatly, and so you can run some cell to confirm it worked (if needed i will press "Revert" when VS code prompts me if he detects any conflict). 			
					
		Never use some python or other code to do "replace string" like for example replace "my_function_A" with "my_function_B", because you always mess up my code with that. there are always some side effects and unpredictable consequences. Always use you inline edit tool for the respective file type. 			
					
	On Finishing Each Task				
					
					
					
					
					
					
					
					
					
		Then, give a short chat summary: what was implemented, in which file/folder, and what the data flow is. No summary files. Just tell me in chat.			
					
	Take your time. Quality over speed.				