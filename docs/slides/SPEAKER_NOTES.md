# Socrates — Loom recording: speaker notes

**Total length target:** under 5 minutes. Aim for ~25–30 sec per slide.
**Audience:** non-technical (PMs, execs, friends from outside ML).

**How to record:**
1. Open `socrates-slides.html` in a browser. Press `F` for fullscreen.
2. Open Loom. Pick "Screen + Cam" (or just Screen if you'd rather not be on camera).
3. Hit record. Read or paraphrase the notes below. Don't memorize — paraphrasing sounds better.
4. Advance with `→` arrow key.
5. Stop. Loom will give you a shareable link.

**Tone:** conversational, like you're explaining this to a smart friend over coffee. The slides carry the data; the voiceover carries the *why*.

**When you don't know what to do with your hands:** keep them off the keyboard between slide transitions so the click sound doesn't bleed in. Pause for half a second before advancing.

---

## Slide 1 — Title (~20 sec)

> Hi — I'm going to walk you through Socrates, a project we did at Hexo. The short version: we figured out that AI research agents — the systems that can write code, run experiments, all of that — fail at things they obviously know how to do. And we found a really simple fix. A second agent that's only allowed to ask questions. No answers, no advice, just questions. That alone gives us a 56% average improvement on Kaggle competitions. Let me show you why.

## Slide 2 — The puzzle (~30 sec)

> Here's the puzzle that started this project. If you give a frontier language model an exam — multiple choice, short answer, all of it — it scores above 88% on machine learning methodology. It knows cross-validation, it knows about data leakage, it knows about overfitting. It can teach a class on this stuff.
>
> Now turn that same model loose as an autonomous agent and ask it to actually do a Kaggle competition. The best result on a benchmark called MLE-bench is a bronze medal in 16.9% of competitions. So the model that gets an A on the exam fails at the bench. Why?

## Slide 3 — Our claim (~35 sec)

> Our claim is that the bottleneck is what we call *knowledge activation*. The agent has the textbook in its head — it just doesn't crack it open at the right moment. Like a graduate student who knows all the right things to do but freezes when their advisor walks out of the lab.
>
> Now you might think — there are already fixes for this, right? You can have the AI critique its own work. That doesn't work, because the same context that made the mistake is now critiquing it. You can have multiple AIs debate. That doesn't work either, because they all share the same blind spots. You can have a manager AI tell a worker AI what to do. That just transfers the manager's blind spots to the worker. So we tried something different.

## Slide 4 — Protocol (~30 sec)

> Here's what we built. On the left, the protocol. We have a Scientist agent — that's the one that actually does stuff, writes code, runs experiments. And we have a second agent we call Socrates, after the Greek philosopher who taught by asking questions.
>
> Before every experiment, the Scientist writes a plan: here's my hypothesis, here's my method, here's what I expect to see. Socrates reads it and only asks questions. The Scientist can't run anything until Socrates says "approved." On the right is what this actually looks like — Socrates asks "is your strategy oriented toward closing the gap?" and the Scientist admits it's been avoiding deep learning, pivots, and the score jumps 10%.

## Slide 5 — The Socratic constraint (~30 sec)

> The reason this works is the constraint. Socrates is not allowed to give answers. Can't say "use 5-fold cross-validation." Can't issue directives. Can't say "try a different model." And it has no tools — no code, no file access. Only questions.
>
> Why is that important? Because if Socrates *could* tell the Scientist what to do, we'd be measuring how good Socrates is. By forbidding it, we force any improvement to come from the Scientist's own knowledge — the textbook that was already in its head. Socrates just makes it crack the textbook open.

## Slide 6 — Conditions (~25 sec)

> We tested three conditions, all using the same underlying LLM. Scientist alone, with no supervision — that's our baseline. Then a "Baseline PI" that's a second agent in the same setup, matched on tokens and turns, but it only gives generic encouragement — "please keep iterating." And then full Socrates. The middle one is critical because it isolates the effect of *what* the advisor says from the effect of just having a second agent in the room.

## Slide 7 — Results (~35 sec)

> Here are the numbers. Five Kaggle competitions spanning radar imagery, RNA biology, GPS, NFL tracking, and ventilator pressure modeling. Socrates gets the best test score on 4 out of 5 tasks. The smallest improvement is 4.8%, the biggest is 195%, and the average is +55.9%.
>
> Crucially, Socrates also beats the Baseline PI on 4 out of 5. So the gain isn't just "having a second agent helps" — it's specifically the structured questioning.

## Slide 8 — Mechanisms (~40 sec)

> A human reviewer read every single experiment log and dialogue transcript to figure out *why* this works. Four mechanisms recur.
>
> One — it catches methodological errors. On the COVID task, Socrates asked one question that revealed the agent had 512 useless features out of 963. The Scientist alone never noticed.
>
> Two — it forces diversification. On Statoil, the Scientist alone ran 16 experiments, 12 of which were the same model with different hyperparameters. With Socrates, the same agent tried 9 completely different model families.
>
> Three — and this one's clever — investigations happen *during* the review. The Scientist still has tools while talking to Socrates. So when Socrates asks "how many features have zero importance?" the Scientist runs the analysis right there.
>
> Four — the approach evolves. On NFL it expanded the feature set. On Ventilator it shrank it. On COVID it regularized. Same advisor, opposite directions — the Scientist retrieves the relevant knowledge for the task.

## Slide 9 — When it fails & variance (~25 sec)

> Honest moment: Socrates loses on one of the five tasks, Ventilator. Why? That task rewards running lots of experiments fast, and Socrates' careful review adds overhead. So if methodology isn't the bottleneck and pure volume is, generic encouragement wins.
>
> On the right — could all this be noise? LLM agents are high variance. We re-ran the Scientist-only condition 10 times on Smartphone with different random seeds. The standard deviation was 15% of the mean. Socrates landed below mean-minus-one-standard-deviation. So not seed noise alone.

## Slide 10 — Take-away (~25 sec)

> Putting it all together: the bottleneck for autonomous AI agents isn't what they don't know. It's what they don't *retrieve* when it matters. A question-only advisor — cheap, simple, no fancy machinery — is enough to bridge that gap on most tasks.
>
> Code is up on GitHub, blog post on my personal site, paper is at COLM. Thanks for watching.

---

## If you want to make it shorter

To get under 4 minutes, the safest cuts are:
- Combine slides 4 and 5 (skip the standalone constraint slide, mention the three forbidden things while showing the protocol diagram).
- Drop slide 6 (conditions) — just mention "we compared the full protocol against the Scientist alone and against a generic-encouragement control" while transitioning into slide 7.
- Shorten slide 8 to two mechanisms instead of four.

## Tone notes (mistakes I make when I'm narrating)

- **Don't say "as you can see"** — the viewer can see it. Just say what's there.
- **Don't read the slide word-for-word** — the slide is the visual aid, your voice carries the *why*. If a slide says "+55.9%" you don't have to repeat the number; you can say "the average gain is around 56 percent."
- **Pause before advancing.** A half-second of silence reads as confidence. Rushing to the next slide reads as nervous.
- **If you flub a sentence, just keep going.** Loom edits later; or don't — rough cut is the goal.
