#!/usr/bin/env python3
"""Weekend jobs: market analysis + strategy evolution."""
from bot.weekend_analyzer import WeekendAnalyzer
from bot.strategy_evolver import StrategyEvolver
from bot.agents.post_mortem import PostMortemAnalyzer

print('=== Weekend Intelligence Report ===')
analyzer = WeekendAnalyzer()
analyzer.run_full_analysis()

print('\n=== Post-Mortem Analysis (Weekly) ===')
pm = PostMortemAnalyzer()
pm.run_daily_postmortem()

print('\n=== Strategy Evolution ===')
evolver = StrategyEvolver()
result = evolver.run_evolution()
if result:
    print(f'Best evolved formula: {result}')
else:
    print('Strategy evolution skipped (gplearn not installed)')

print('\nAll weekend jobs complete!')
