# TINY LLM — build, train, test and serve your own small language model

> This README documents **LLM-CREATION-PROGRAM** (`TINY LLM PROGRAM VERSION 9.py`).
> If you use a different copy of the program, a few prompts or defaults may differ.
> There are 9 versions of this program this is the one with least issues others are not published in GitHub due to error in software/code.
---

## Contents

1. [What the program is](#1-what-the-program-is)
2. [Features at a glance](#2-features-at-a-glance)
3. [Requirements and installation](#3-requirements-and-installation)
4. [Starting the program](#4-starting-the-program)
5. [Quick start: train your first model](#5-quick-start-train-your-first-model)
6. [Main menu reference](#6-main-menu-reference)
7. [How it works](#7-how-it-works)
8. [Input file formats](#8-input-file-formats)
9. [Files the program creates](#9-files-the-program-creates)
10. [Using a GPU](#10-using-a-gpu)
11. [Troubleshooting](#11-troubleshooting)
12. [Limitations and good-to-know](#12-limitations-and-good-to-know)
13. [Glossary](#13-glossary)
14. [Version history](#14-version-history)

---

## 1. What the program is

TINY LLM is a single-file Python program (built on PyTorch) that lets you create a **small GPT-style language model from scratch**, using nothing but your own text, and then look after that model through a simple text menu. From the menu you can:

- **train** a new model on text files,
- **keep training** it later, on the same data or new data,
- **improve it with feedback**, either from a larger model running in [Ollama](https://ollama.com) or from a Q&A file,
- **measure** how good it is (loss, perplexity, accuracy, validation, a final test, quizzes),
- **chat with it** and see generated text,
- **serve it** through an Ollama-compatible API so other programs can talk to it, or export it as a package.

You do not need to write code. Everything is menu-driven, and every long task can be paused and resumed.

**What to expect.** The default model has about **166,000 parameters** (with the default 512-token vocabulary), which is tiny compared with commercial models. It runs on an ordinary laptop. It learns the vocabulary, style and patterns of the text you give it. It is a great way to learn how language models work and to experiment, but it will not "know" things the way a large model does.

---

## 2. Features at a glance

**Training**
- Byte-level **BPE tokenizer** trained on your own text (vocabulary size you choose, 256 minimum, 512 default).
- Decoder-only **transformer** with causal self-attention (default: 64 wide, 2 layers, 4 heads).
- **AdamW** optimizer with adjustable weight decay (applied to weight matrices only).
- **Learning-rate schedule**: warm-up, then cosine decay.
- **Automatic backups** every N epochs; **pause** any time and **resume** later without re-entering settings.
- Optional **early stopping** when a target loss is reached, with an offer to train extra epochs if it is not.
- A live one-line progress display with loss, learning rate, time per epoch and ETA.
- **Loss-per-epoch graph** (PNG) and a **CSV of the loss data**, saved automatically for later analysis.
- Uses your **NVIDIA GPU automatically** when PyTorch can see one.

**Fine-tuning and feedback**
- Train more epochs on old or new data.
- **Ollama feedback learning**: a bigger model in Ollama answers your questions; your model learns from those answers.
- **Reinforcement-style feedback learning**: your model answers, an Ollama model scores each answer 0–10, and low-scoring answers are used to improve the model.
- **Conversational training** from a Q&A text file (no Ollama needed).

**Evaluation**
- **Score** a model on any dataset without training (loss, perplexity, accuracy, rating).
- **Validate** (repeatable overfitting check with history) and **Test** (one-shot final benchmark with a certificate).
- **Multiple-choice quizzes** with a clear right/wrong score.
- Optional **question-file scoring** after training or chatting.

**Using the model**
- Interactive **chat/generation** with optional automatic scoring by Ollama.
- **Ollama-compatible server** (`/api/generate`, `/api/chat`, …).
- **Export package** with weights, tokenizer, a Modelfile and a launcher script.

---

## 3. Requirements and installation

### 3.1 Python and packages

| What | Needed? | Used for |
|---|---|---|
| **Python 3.10 or newer** (developed on 3.12) | Required | Running the program |
| **torch** (PyTorch) | Required | The model, training, GPU support |
| **numpy** | Required | Numerical helpers |
| **requests** | Required | The program imports it at start-up; used only to talk to Ollama on your own computer |
| **pillow** | Required | Imported at start-up |
| **matplotlib** | Optional, recommended | The loss graph. Without it, training works and the CSV is still saved |
| **Ollama** (separate app) | Optional | Needed only for options 2-2, 2-3 and the optional automatic scoring (see section 6) |
| PyPDF2, pytesseract | Not needed | Only used by reference-file code that is not connected to any menu option in this version |

Install the packages from a Command Prompt or terminal:

```
pip install numpy requests pillow matplotlib
```

PyTorch is installed separately because there are CPU and GPU builds. Use the selector on <https://pytorch.org/get-started/locally/> and run the command it gives you. Pick a **CUDA** build if you have an NVIDIA GPU.

> Tip: if you have several Python versions, use `python -m pip install ...` (or `py -m pip install ...` on Windows) so the packages go into the same Python you run the program with.

### 3.2 Ollama (only for some features)

[Ollama](https://ollama.com) runs larger language models on your computer. The program talks to it at `http://localhost:11434`. Install Ollama, download at least one model (for example `ollama pull llama3.2`), and keep Ollama running while you use options 2-2 and 2-3, or the "use Ollama to score" choices.

---

## 4. Starting the program

- **IDLE:** open `TINY_LLM_PROGRAM_VERSION_7_9_1.py` and press **F5**.
- **Terminal / Command Prompt:** `python TINY_LLM_PROGRAM_VERSION_7_9_1.py`

Start it from the folder where you want your files kept: the program creates a **`saved_models`** folder in the *current working folder* (see section 9).

On start-up you see the banner and which device will be used:

```
TINY LLM — INTERACTIVE INTERFACE
Device: cuda        (or: Device: cpu)
```

Then the main menu appears.

---

## 5. Quick start: train your first model

1. **Get some text.** Put your text in a plain `.txt` file (UTF-8), or in a folder of `.txt`/`.md` files. More text gives better results: a few kilobytes will only be memorised, hundreds of kilobytes or more start to show real style.
2. Start the program and choose **1 — Train a new model**.
3. When asked for data, add your file or folder path. Press Enter on an empty line when you are done adding paths. (The paths are remembered for next time.)
4. Type a **model name**, for example `story_model`.
5. Answer the settings prompts. **Pressing Enter on each one accepts the defaults**, which is fine for a first run:

   | Prompt | Default |
   |---|---|
   | Epochs | 10 |
   | Learning rate | 0.001 |
   | Weight decay | 0.01 |
   | Batch size | 4 |
   | Backup every N epochs | 5 |
   | BPE vocabulary size | 512 |
   | Early-stop target loss | (blank = off) |

6. Watch the progress line. To stop and save at any time, press **Ctrl+C** or type **`q`** and press Enter (see section 6, option 1).
7. When training ends you get a summary and the paths of the loss graph and data file.
8. Back at the main menu, choose **3 — Load and generate**, pick `story_model`, and type a prompt.

> On a GPU, use a **larger batch size** (for example 64 or 128) to benefit from the speed-up. See [section 10](#10-using-a-gpu).

---

## 6. Main menu reference

```
MAIN MENU
  1. Train a new model
  2. Fine-tune existing model
  3. Load and generate with existing model
  4. Export / Serve model for Ollama
  5. List saved models
  6. Exit
```

Anywhere a prompt shows a default in brackets, **pressing Enter accepts it**.

### Option 1 — Train a new model

Creates a new model and trains it on your text.

**Step 1: choose the data.** The program shows the dataset paths you used before (separately for training data) and asks you to choose:

| Choice | Meaning |
|---|---|
| a number, or several such as `1 3` | use those saved datasets |
| `a` | use all saved datasets |
| `n` | skip the saved ones and add new paths |
| `m` | manage the saved list (add or remove paths) |

After choosing saved datasets it asks *"Add more paths on top? (y/n)"*. A path can be a **`.txt` file** or a **folder** (all `.txt` and `.md` files inside it, including sub-folders, are read). All chosen sources are joined into one training text.

**Step 2: model name.** Default `default_model`. If a model with that name has an **unfinished training session**, you see its saved settings and the question *"Reload all saved parameters and resume? (y/n)"*. Answer `y` to continue exactly where you stopped (no further questions are asked). If only a backup file exists, you are asked *"Backup found … Resume? (y/n)"*.

**Step 3: settings.**

| Prompt | Default | What it does |
|---|---|---|
| Epochs | 10 | How many full passes over your text |
| Learning rate | 0.001 | The peak learning rate (see the schedule in section 7.4) |
| Weight decay | 0.01 | AdamW regularisation. `0` switches it off. Applied to weight matrices only |
| Batch size | 4 | How many text windows are trained on at once. Bigger is faster on a GPU |
| Backup every N epochs | 5 | How often a resume checkpoint is written |
| BPE vocabulary size | 512 (minimum 256) | Size of the tokenizer's vocabulary |
| Early-stop target loss | none | Training stops automatically once an epoch's loss is at or below this |

**What happens next.** The program trains a BPE tokenizer on your text, builds the model, cuts the text into overlapping 32-token windows, saves the session so it can be resumed, and starts training. If your text is too short it tells you (it needs at least 33 tokens) and you can use more text or a smaller vocabulary.

**The training screen.** Before the first epoch you see a banner (epochs, optimizer, where the loss log is saved, the learning-rate schedule and how to stop). Each epoch then updates one line:

```
Epoch  3/10  [█████████░░░░░░░░░░░░░░░░░░░░░]  30.0%  Loss: 1.2345  LR: 1.00e-03  9s/ep  ETA: 1m 03s
```

**Stopping and pausing.**

| Action | Effect |
|---|---|
| **Ctrl+C** | Stops almost immediately (after the current batch), saves a backup and pauses |
| Type **`q`** then Enter (`quit`, `s`, `stop` and `x` also work) | Stops cleanly **after the current epoch**, saves a backup and pauses |

In **IDLE** the `q` key is not available (the program tells you so); use **Ctrl+C**.

**Resuming.** Run option 1 again with the **same model name** and answer `y` to resume (or use option 2 → 1, below). All settings are restored automatically. (The on-screen hint that mentions "Option 8" is out of date; use option 1 or option 2 → 1.)

**If the target loss is not reached.** When you set an early-stop target and training finishes without reaching it, the program offers *"Extra epochs to train:"*. Type a number to continue the same session for that many more epochs, or press Enter to stop and keep the saved model.

**After training** you get a summary box (model size, vocabulary, epochs trained, final loss), and the files described in section 9.

---

### Option 2 — Fine-tune existing model

A sub-menu for everything you do to a model that already exists:

```
  1. Train more epochs on new or old data
  2. Ollama model-based feedback learning
  3. Use already-trained LLM for reinforcement feedback learning
  4. Score model on dataset        (no training - pure evaluation)
  5. Validate model                (repeatable - overfitting check)
  6. Test model                    (one-shot final benchmark)
  7. Train on conversational dataset file (no Ollama needed)
  8. Multiple-choice quiz
  9. Back
```

You choose the model from a numbered list; models with an unfinished session are marked so you can resume them.

#### 2-1 Train more epochs on new or old data

Continues training an existing model.

1. Pick the model. If it has an unfinished session you can resume it (*"Resume from saved parameters? (y/n)"*).
2. Otherwise choose *"Load new training data? (y = new file, n = use previous)"*. With `y` you pick data as in option 1; with `n` the previous dataset is used.
3. Settings: **Extra epochs** (10), **Learning rate** (0.001), **Weight decay** (0.01), **Batch size** (4), **Backup every N epochs** (5), **Early-stop target loss** (none).
4. Training runs exactly as in option 1 and starts a new session in the loss history (see 7.5).
5. Afterwards you are offered the optional question-file scoring (see "Question-file scoring" below).

#### 2-2 Ollama model-based feedback learning

A bigger model in Ollama acts as a teacher. Your questions are answered by Ollama, and your small model is fine-tuned on those question-and-answer pairs. **Needs Ollama running.**

1. Choose which Ollama model should write the answers.
2. Optionally enter extra instructions for the teacher (for example "answer in one short sentence").
3. Enter the settings up front: **Epochs** (10), **Batch size** (2), **Learning rate** (0.0005), **Weight decay** (0.01), **Save checkpoint every N epochs** (2), **Early-stop loss** (none). A summary is shown and you confirm with `y`.
4. Give a **question file** in the `@question@` format (section 8.2). The file is checked first.
5. Ollama answers every question. The answers are logged to `<model>_ollama_answers_<timestamp>.log`, and the training text is saved as `generated_training_<timestamp>.txt` in `saved_models`.
6. Your model is fine-tuned on that text, then you may run the optional question-file scoring.

After the settings are confirmed, the rest runs without further prompts.

#### 2-3 Reinforcement feedback learning (an LLM as the teacher)

A live loop where **your model answers, Ollama scores the answer from 0 to 10, and weak answers are used to improve the model.** **Needs Ollama running.**

Setup questions:
- Which Ollama model should judge the answers.
- Optional extra instructions for the judge.
- *"Pause when the average score reaches (0–10, Enter to skip)"*: a target average.
- Where the questions come from: **1 — type them yourself**, or **2 — load them from a folder** of `.txt`/`.md` files (format in section 8.5). If you used a folder before, you are asked whether to use it again. With a folder you choose **1 — manual** (you press a key for each next question) or **2 — automatic** (it runs by itself).

Commands inside the loop:

| Command | What it does |
|---|---|
| type a question | the model answers (prompt format `Q: … A:`), Ollama scores it |
| `next` | go to the next question from the folder |
| `manual` | switch from automatic to manual mode |
| `feedback` | show statistics: number of answers, average, best, worst, how many were good (6 or more) and how many need improvement, plus the low-scoring answers |
| `improve` | fine-tune the model on the collected feedback. You choose the number of epochs (5–15 is recommended). The model is saved afterwards |
| `quit` | print the final statistics and leave |

How scores are read: **8–10 excellent**, **6–7.9 good**, **below 6 needs improvement** (the answer is added to the learning set). When the running average reaches your target, the loop says **TARGET REACHED**: press Enter to keep going, or type `quit` to save the model and exit.

> `quit` on its own does **not** save the model. The model is saved by `improve` or when you quit after the target is reached.

#### 2-4 Score model on dataset (no training)

Measures how well the model predicts a dataset, **without changing the model.**

- Pick data as in option 1 (it uses the same saved-paths list as training).
- The text is converted to the model's vocabulary (a coverage percentage is shown).
- Prompts: **Sequence length** (32) and **Batch size** (4).
- The report shows **loss**, **perplexity**, **next-token accuracy** and the spread of per-batch loss, with a **rating** (section 7.7). It is saved as `<model>_score_<timestamp>.txt`.

#### 2-5 Validate model (repeatable overfitting check)

Run this as often as you like on a **separate validation dataset** (kept in its own saved-paths list) to see whether the model generalises.

- Reports loss, perplexity, accuracy and rating.
- Compares the validation loss with the model's last training loss and tells you whether it looks **healthy, mildly overfitted, strongly overfitted or underfitted**.
- Keeps a **history** of validation runs (the last five are shown with trend arrows) in `<model>_val_history.json`, and saves each report as `<model>_val_<timestamp>.txt`.

#### 2-6 Test model (one-shot final benchmark)

Run this **once**, on a **separate test dataset** that the model has never been trained or validated on (it has its own saved-paths list).

- Because a test is meant to be a one-time result, the program warns you if a test report already exists and asks *"Run the test anyway? (yes/n)"*.
- Shows sample predictions against the true next tokens.
- Writes a detailed **test certificate** to `<model>_TEST_CERTIFICATE_<timestamp>.txt`.

#### 2-7 Train on conversational dataset file (no Ollama needed)

Fine-tunes the model on a file of questions and answers you wrote yourself.

1. Pick the model, then enter the settings up front: **Epochs** (10), **Batch size** (4), **Learning rate** (0.001), **Weight decay** (0.01), **Backup every N epochs** (5), **Early-stop target loss** (none). You confirm the summary.
2. Choose the dataset with the same chooser as option 1; conversational files have their own saved-paths list.
3. The text is converted to the model's vocabulary, the session is saved, and training runs. If you stop it, run option 2-7 on the same model again to resume.
4. The optional question-file scoring is offered at the end.

The recommended file layout is `Q:` / `A:` blocks (section 8.3), but any text works.

#### 2-8 Multiple-choice quiz

An objective way to score the model: it must choose the right option from a fixed set.

A sub-menu:

```
  1. Load / select question file(s)
  2. Add a question to current file
  3. Delete a question from current file
  4. Create a new question file
  5. Run quiz — ask the LLM all loaded questions
  6. Back to fine-tuning menu
```

- **1** picks saved quiz files (a number, several numbers, `a` for all, a new path, or `m` to manage the list).
- **2 / 3** add or delete a question in the current file. **4** creates a new file (a bare name is saved in `saved_models/mcq_datasets/`).
- **5** asks the model every loaded question. Each question shows the correct answer plus up to three wrong ones, in random order. The model's reply is read as a choice (an option number such as `2` or `Option 2`, or the option text). The result shows the percentage correct, how many replies could not be understood, and a rating: **80 % or more excellent**, **60 % good**, **40 % fair**, **20 % poor**, otherwise **very poor**. A log is saved as `<model>_mcq_quiz_<timestamp>.log`.

The file format is in section 8.4.

#### 2-9 Back

Returns to the main menu.

#### Question-file scoring (offered after several actions)

After options 2-1, 2-2, 2-3 and 2-7, and after chatting in option 3, the program may ask *"Score the model using a question file? (y/n)"*. If you say yes:

- Give a text file of questions (one per line).
- Set the **minimum average score to pass** (0–10, default 6) and the **extra epochs** to train if the model falls below it (default 10).
- Choose whether **Ollama scores the answers automatically** or you rate each answer yourself (0–10, Enter to skip).
- If the model does not reach the pass mark, the program offers to retrain it.

---

### Option 3 — Load and generate with existing model

Chat with a trained model.

1. Pick the model.
2. Choose *"Use Ollama to check/score generated answers? (y/n)"*. If yes (and Ollama is running) you may add scoring instructions.
3. Type prompts at the **`Prompt:`** line. The model replies with a short answer. With Ollama scoring on, each reply also gets a 0–10 score with a comment.
4. Type **`quit`** to finish. The optional question-file scoring is then offered.

Generation settings (fixed in the program): up to 200 tokens, temperature 0.85, top-k 10, a repetition penalty of 1.5, and the answer may end at a sentence end (`.`, `!`, `?` or a blank line) once it is at least 15 characters long.

---

### Option 4 — Export / Serve model for Ollama

```
  1. Start local Ollama-compatible server
  2. Export model package
  3. Back to main menu
```

#### 4-1 Start local Ollama-compatible server

Lets other programs use your model through the same API style as Ollama.

- Prompts: **Port** (default **11435**; real Ollama uses 11434), **Temperature** (0.8) and **Max tokens** (200).
- Press **Enter** in the console to stop the server.
- Endpoints: `GET /api/tags` (list the model), `GET /api/version`, `POST /api/show`, `POST /api/generate` and `POST /api/chat` (streaming by default, or non-streaming with `"stream": false`).
- To use it from this program's Ollama features, set the Ollama URL to `http://localhost:11435`. Other apps: point their Ollama host to `localhost:11435`.

> **Security note.** The server listens on **all network interfaces** and has **no password**, so other devices on your network could reach it. Windows may show a firewall prompt: allow it only on a network you trust, or close the server when you are done.

#### 4-2 Export model package

Writes a folder (default `ollama_export_<model name>`) containing:

| File | Purpose |
|---|---|
| `<model>.pt` | the trained weights |
| `<model>_tokenizer.json` | the tokenizer |
| `Modelfile` | an Ollama-style description of the model |
| `serve.py` | a launcher that starts the server for this model |
| `README.txt` | short instructions |

`serve.py` loads **this program file** as a library, so keep the program file in place (the launcher points to where it was when you exported).

---

### Option 5 — List saved models

Prints the names of all models saved in `saved_models`.

### Option 6 — Exit

Closes the program.

---

## 7. How it works

### 7.1 The tokenizer

Models do not read letters; they read **tokens**. Option 1 trains a **byte-level BPE** tokenizer (like the one used by GPT-2) on *your* text. It starts from all 256 byte values and repeatedly merges the most common pairs until the vocabulary reaches the size you chose. A larger vocabulary gives shorter token sequences but more parameters in the model's input and output layers. The tokenizer is saved as `<model>_tokenizer.json` and reused for all later work on that model.

### 7.2 The model

A small decoder-only transformer, the same family as GPT:

- token embedding + fixed (sinusoidal) position information,
- *N* transformer blocks, each with **causal multi-head self-attention** and a **feed-forward** layer (width 4 × the model width, ReLU), with residual connections and layer normalisation,
- a final linear layer that produces a score for every token in the vocabulary.

The menus use **64 wide, 2 blocks, 4 attention heads, dropout 0.1**, and a 100-token position table (longer inputs are handled by extending the position information automatically). This is fixed in the menus; changing it means editing the `SimpleLLM(...)` lines in the code.

### 7.3 How text becomes training examples

The token stream is cut into overlapping **windows of 32 tokens**. For each window the model sees tokens 1…32 and must predict tokens 2…33. One **epoch** is one pass over all windows; the **batch size** is how many windows are processed together.

### 7.4 Optimizer and learning-rate schedule

- **AdamW** with the weight decay you chose (default 0.01). Decay is applied to weight matrices only, never to biases or layer-norm values.
- The learning rate **warms up** from near zero to your chosen value over the first 10 % of the epochs (at least one epoch), then follows a **cosine curve down to 10 % of the peak**. The training banner shows the exact epoch ranges.

### 7.5 Loss graph and loss data

Every finished epoch is recorded automatically:

- **`<model>_loss_history.csv`** has one row per epoch with the columns `session, epoch, loss, learning_rate, weight_decay, epoch_seconds, logged_at`. It opens directly in Excel or pandas. All sessions of a model are kept in the one file.
- **`<model>_loss_curve_session<N>.png`** is the graph: **epoch on the x axis, loss on the y axis**, with the lowest loss marked and the target-loss line drawn if you set one. It is redrawn at every backup and when training ends.

A **session** is one training run: starting a fresh training (or a new fine-tune) begins a new session; pausing and resuming continues the same one. Epochs cut short by Ctrl+C are not recorded, so they appear as a gap.

Settings at the top of the "LOSS HISTORY" section of the program file:

```
LOSSPLOT_ENABLED     = True    # False = record nothing and draw nothing
LOSSPLOT_SHOW_WINDOW = False   # True  = also pop the graph up in a window when training ends
```

If matplotlib is not installed, or the CSV is open in another program (Excel locks files), training carries on: the program prints a warning, keeps the data and writes it when it can.

### 7.6 Backups, pausing and resuming

- A **backup** (`<model>_backup.pt`) holds the weights, the optimizer state, the schedule position and the epoch. It is rewritten every *N* epochs and whenever you stop.
- A small **session file** (`<model>_train_state.json`) stores your settings and progress so a resume needs no re-typing.
- When training finishes normally the final weights are saved as `<model>.pt`.
- Backups made by older program versions (plain Adam) can still be resumed. The weights and schedule are restored and the optimizer's running averages start fresh, which the program tells you.

### 7.7 Reading the evaluation numbers

| Number | Meaning |
|---|---|
| **Loss** | Average cross-entropy on the next token. Lower is better |
| **Perplexity** | `exp(loss)`: roughly "how many tokens the model is torn between". Lower is better |
| **Accuracy** | How often the model's top guess is the exact next token |

Rating from perplexity: **below 5 excellent**, **below 15 good**, **below 50 fair**, **below 200 poor**, otherwise **very poor**.

The validation check compares validation loss with training loss. A validation loss far above the training loss (a gap over about 1.5 is flagged as strong overfitting, over about 0.5 as mild) means the model is memorising instead of learning.

---

## 8. Input file formats

### 8.1 Training text

A plain UTF-8 `.txt` file, or a folder whose `.txt` and `.md` files are all read (sub-folders included). Several sources can be combined.

### 8.2 Question file for option 2-2 (`@` format)

Each question is wrapped in `@` signs. Use blank lines between them:

```
@What is the capital of France?@

@Explain what a noun is in one sentence.@
```

The file must contain an even number of `@` signs; the program checks this and tells you how many questions it found.

### 8.3 Conversational file for option 2-7

Recommended layout:

```
Q: What is a noun?
A: A noun is a word that names a person, place or thing.

Q: What is a verb?
A: A verb is a word that describes an action.
```

### 8.4 Multiple-choice file for option 2-8

One question per line, separated by `|`: the question, the **correct** answer, then one or more **wrong** answers.

```
What is the capital of France?|Paris|London|Berlin|Madrid
Which gas do plants absorb?|Carbon dioxide|Oxygen|Helium
```

At least one wrong answer is required. When the quiz runs, up to three wrong answers are picked at random and mixed with the correct one.

### 8.5 Question folder for option 2-3

A folder of `.txt` or `.md` files (sub-folders included). In each file, the program takes:

1. every line that ends with `?`; if there are none,
2. numbered items (`1. …`, `2. …`); if there are none,
3. the first sentence of each paragraph (turned into a question).

### 8.6 Question list for question-file scoring

A text file with one question per line.

---

## 9. Files the program creates

Everything lives in a **`saved_models`** folder inside the folder you started the program from.

| File | What it is |
|---|---|
| `<model>.pt` | trained weights (saved on the CPU, so they load on any machine) |
| `<model>_backup.pt` | resume checkpoint (weights, optimizer, schedule, epoch) |
| `<model>_tokenizer.json` | the model's tokenizer |
| `<model>_train_state.json` | settings and progress of the training session |
| `<model>_loss_history.csv` | per-epoch loss data, all sessions |
| `<model>_loss_curve_session<N>.png` | the loss graph for session N |
| `<model>_training_log.txt`, `<model>_qa_log.txt`, `<model>_metadata.json` | training and feedback logs |
| `<model>_config.json` | remembered preferences, for example your question folder |
| `<model>_score_<time>.txt` | report from 2-4 |
| `<model>_val_<time>.txt`, `<model>_val_history.json` | reports and history from 2-5 |
| `<model>_TEST_CERTIFICATE_<time>.txt` | certificate from 2-6 |
| `<model>_mcq_quiz_<time>.log` | quiz log from 2-8 |
| `<model>_ollama_answers_<time>.log`, `generated_training_<time>.txt` | from 2-2 |
| `dataset_registry.json` | saved training/scoring data paths |
| `validation_dataset_registry.json`, `test_dataset_registry.json` | saved validation and test paths |
| `conversational_dataset_registry.json`, `mcq_dataset_registry.json` | saved conversational and quiz paths |
| `mcq_datasets/` | quiz files created from the menu |

Exported packages go to `ollama_export_<model>/` (option 4-2).

To **back up or move a model**, copy `<model>.pt` and `<model>_tokenizer.json`. To keep a resumable session too, also copy `<model>_backup.pt` and `<model>_train_state.json`.

---

## 10. Using a GPU

- The program uses an **NVIDIA GPU automatically** when your PyTorch build can see one. The start-up banner shows `Device: cuda` (GPU) or `Device: cpu`.
- If it says `cpu` on a GPU computer, you most likely have the CPU-only build of PyTorch. Reinstall from <https://pytorch.org/get-started/locally/> and choose a CUDA build. You can check with:
  `python -c "import torch; print(torch.cuda.is_available())"`
- **Batch size matters.** The tiny default model keeps a GPU busy only with bigger batches. On a laptop RTX 3050 the GPU was about 1.5x faster than the CPU at batch size 32 and about 6x faster at batch size 256 (and 8–11x for a larger model). Try 64–256 when you train on a GPU. Larger batches mean fewer updates per epoch, so you may want to adjust the learning rate. Real training is a little slower than these figures because loading data also takes time.
- Weight files are saved as CPU tensors, so a model trained on a GPU also loads on a computer without one.
- A separate helper script, **`gpu_smoke_test.py`**, checks your GPU set-up and times CPU against GPU. It does not install anything.

**Windows note: "Smart App Control".** On some Windows 11 PCs, Smart App Control blocks PyTorch's DLL files with the error *"An Application Control policy has blocked this file"* (WinError 4551). It can also start blocking days after a new PC is set up. See the troubleshooting table.

---

## 11. Troubleshooting

| Problem | What to do |
|---|---|
| `ModuleNotFoundError: No module named 'requests'` (or `PIL`, `numpy`, …) | Install the package into the Python you run the program with, for example `python -m pip install requests pillow` |
| **WinError 4551 — "An Application Control policy has blocked this file"** when importing torch | Windows Smart App Control is blocking unsigned DLLs. Check *Settings → Privacy & security → Windows Security → App & browser control → Smart App Control settings*. Turning it off fixes it (Microsoft Defender antivirus keeps running). Install Windows updates first, because some builds can have trouble switching it back on. The alternative is to run the program inside WSL2 (Linux). |
| `Device: cpu` although you have an NVIDIA GPU | CPU-only PyTorch is installed. Install a CUDA build (section 10) |
| No loss graph (PNG) | matplotlib is missing: `python -m pip install matplotlib`. The CSV is still saved |
| Warning that loss data could not be written | The CSV is open in another program (for example Excel). Close it; the data is kept and written at the next epoch, or saved to a side file at the end |
| "Training text is too short … need at least 33 tokens" | Use more text, or choose a smaller BPE vocabulary |
| Options 2-2 / 2-3 say Ollama is not available | Start Ollama and make sure a model is installed (`ollama list`). The program expects it at `http://localhost:11434` |
| The `q` key does nothing in IDLE | This is expected in IDLE. Use **Ctrl+C** |
| GPU out-of-memory error | Lower the batch size |
| I cannot find my paused session | Use the **same model name** and the **same option** (option 1, or option 2 → 1 / 2-7) and answer `y` to resume |
| The generated text looks like nonsense | Normal for a tiny model with little text or few epochs. Use more text, train longer, and watch the validation loss |
| Firewall popup when starting the server | The server listens on all network interfaces. Allow it only on a trusted network, or cancel |

---

## 12. Limitations and good-to-know

- **Fixed model size in the menus** (64 wide, 2 layers, 4 heads, 32-token windows). Changing it means editing the code.
- **Resuming reloads only the first data source** recorded for the session (and, for a folder, only its top-level `.txt` files). If you trained on several sources, a resume continues with the first one.
- A training run stopped with **Ctrl+C** counts the interrupted epoch as done, but that epoch's loss is not plotted.
- In **2-3**, `quit` does not save the model unless you used `improve` or reached the target.
- The **server has no password** and listens on every network interface.
- The Ollama features need a separately installed and running Ollama; the quality of the feedback depends on the Ollama model you pick.
- The model is small: it imitates the text you give it. Do not expect factual knowledge.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Epoch** | One full pass over the training data |
| **Batch size** | How many examples are processed together in one step |
| **Learning rate** | How big each update step is. Too high is unstable, too low is slow |
| **Weight decay** | A gentle pull that keeps weights small, which reduces overfitting |
| **Loss** | The training error. Lower is better |
| **Perplexity** | `exp(loss)`, an easier-to-read version of the loss |
| **Overfitting** | The model memorises the training text and does worse on new text |
| **Tokenizer / BPE** | Splits text into pieces (tokens); BPE learns the pieces from your data |
| **Checkpoint / backup** | A saved snapshot you can resume from |
| **Session** | One training run (with its pauses and resumes) |
| **Fine-tuning** | Further training of an existing model |
