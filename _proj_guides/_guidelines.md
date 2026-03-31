					
	Before implementing anything, read 				
		C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\PROJECT_PLAN.md			
					
		and also the project guides: 			
			C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\files_as_md\proj_guide_NLPS_-_TREC_2025_BioGen.md		
			C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\files_as_md\prof_guide_professor_notes.md		
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
		Inline comments during the code itself to explain the code. This is a teaching project so i want to have comments to explain all important details of each block of code we do so a reader can understand it easy, BUT WRITTEN AS CONCISE AS POSSIBLE, AND WRITTEN AS CASUAL AND NATURAL LANGUAGE AS POSSIBLE, LIKE IF IT WAS ONE DEV TALKING TO ANOTHER DEV. SO CONCISE BULLET POINTS, NO FORMAL PROSE. NO EMOJIS ANYWHERE!!!			
		Logs: only what's needed to pinpoint errors. Single-line, concise, no emojis (terminal encoding issues).			
					
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
		Each file has its own if __name__ == "__main__" block with tests. Run everything locally on CPU with small data samples first. We will only move to Colab+GPU once local tests pass.			
		Good tests to include in DL projects:			
			Data tests: shapes correct, labels encoded right, no NaNs, class distribution matches CSV		
			Transform tests: output tensor shape, value range [0,1] or normalized		
			Model tests: forward pass with dummy input gives expected output shape		
			Training tests: loss decreases after 1-2 steps on a tiny batch (sanity check)		
			Submission tests: output CSV has correct number of rows, valid class names, correct format		
		And for NLP project think good tests like: 			
			Connection tests:   OpenSearch reachable, index exists, doc count matches expected 		
			Data tests:         topics load correctly, odd/even split is clean, ground truth relevance values in expected scale		
			Retrieval tests:    each strategy returns results, no empty hits, scores decrease monotonically down the ranked list		
			Evaluation tests:   ranx run file format valid, graded relevance passed correctly, P@10 + R@100 + nDCG all computed without errors		
			Generation tests:   answer ≤250 words, ≤3 PMIDs per sentence, all cited PMIDs exist in corpus		
			Agent tests:        planner returns ≥1 sub-topic, ReAct loop terminates, final report has citations		
		Before you run the tests, first, to confirm the code is well implemented, run the tests with a small sample of data. Only when you confirm the code is all 100% well done and correct and well tested, then yes you run the tests with the full data. 			
					
	Before Implementing Any Task				
		Read all relevant existing files — avoid duplication.			
		Think the 3 best options at architecture level (where/how it fits in the project).			
		Think the 3 best options at implementation level (how to write the code locally).			
		Choose the best one — clean, simple, maintainable. No shortcuts, no over-engineering.			
					
	Workflow				
		implement → local test (CPU, small data) → fix → notebook (Colab, GPU)			
		Never update the notebook until the Python files are tested and working.			
					
	Terminal				
		PowerShell — use ; not && for chaining commands.			
		avoid to run commands with things like 2>&1 and | and filters... prefer to have the logs go normally to the terminal directly so i can also see better what is happening 			
		i have alrady a environment set up with anaconda: lets use "cnn (3.10.19)", so don't create new environemnts 			
					
	On Finishing Each Task				
		Add all important details to the project report:			
			Check the code we implemented and all the tests results... and do a deep analysis about our findings and think what things might be interesting and important to add to the final project report. 		
			The report is in the file: 		
				C:\Users\franc\Desktop\NLP_Biomedical_Agent\tasks\report.md	
			And we should follow this structure, so think all the important details, results and analysis that we should add to each section:		
				" [1. Introduction](#1-introduction)
- [2. BioGen NL Agent](#2-biogen-nl-agent)
  - [a. Data Parsing, Indexing, and Search (Phase 1)](#a-data-parsing-indexing-and-search-phase-1)
  - [b. LLM Augmented Generation (Phase 2)](#b-llm-augmented-generation-phase-2)
  - [c. LLM Agentic Patterns (Phase 3)](#c-llm-agentic-patterns-phase-3)
- [3. Evaluation](#3-evaluation)
  - [a. Experimental Setup: Datasets, Metrics, and Protocols](#a-experimental-setup-datasets-metrics-and-protocols)
  - [b. Results and Discussion](#b-results-and-discussion)
- [4. Conclusion](#4-conclusion)
  - [a. Achievements](#a-achievements)
  - [b. Limitations](#b-limitations)"	
			In terms of writing: don't write full text sentences like the final report. Instead write for now only concise bullet points like in a powerpoint presentation, with all the important details and numbers and analysis, but written in a concise manner. Only latter we'll convert them to a final well written text. 		
			In terms of the notes quantity, you can write all the good ideas you think might be good to have in the final report. Don't bother about some page limit. Only latter we will select and organize the bullet points and we filter out the less important according to the space available to the final report.		
			In terms of the results and analysis, as the project evoolve it is normal that our conclusions might change, so for each bullet point mark the timestamp of that note so we can latter see if some result was overide by a latter one. And also mark where in project we have the code to confirm again that result, so it is easy to run things again and confirm current results. 		
		Then, give a short chat summary: what was implemented, in which file/folder, and what the data flow is. No summary files. Just tell me in chat.			
					
	Take your time. Quality over speed.				