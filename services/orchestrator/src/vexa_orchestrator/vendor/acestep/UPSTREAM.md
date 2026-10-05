# ACE-Step constrained metadata decoding

Source: https://github.com/ace-step/ACE-Step-1.5
Revision: ca1e85fe9430179831e6bc6be790c332190a3866
License: MIT (see LICENSE).

Only constants and the metadata logits processor are included. Changes: relative
constants import and Python standard logging in place of Loguru. The text adapter
updates FSM state from Hugging Face generated tokens, uses understand phase and
injects requested language, duration and meter. No audio generation implementation.
