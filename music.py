import sounddevice as sd
import numpy as np
import threading
import time
import random
import soundfile as sf

from mutagen.wave import WAVE
from mutagen.id3 import ID3, TIT2, TPE1, TALB

from pedalboard import Pedalboard, Reverb, Delay, Chorus, LowpassFilter, HighpassFilter, Resample
import pedalboard

chords = {
    "C": [261.63, 329.63, 392.00, 523.25],  # C E G C
    "Am": [220.00, 261.63, 329.63, 440.00], # A C E A
    "F": [174.61, 261.63, 349.23, 523.25],  # F C F C
    "G": [196.00, 246.94, 392.00, 493.88],  # G B G B
}

active_threads = 0
MAX_THREADS = 8

recording_buffer = []
recording_lock = threading.Lock()

mix_buffer = np.zeros((0, 2), dtype=np.float32)
mix_lock = threading.Lock()

def addToMix(audio, start_sample):
    global mix_buffer
    with mix_lock:
        end_sample = start_sample + len(audio)
        if end_sample > len(mix_buffer):
            padding = np.zeros((end_sample - len(mix_buffer), 2), dtype=np.float32)
            mix_buffer = np.concatenate([mix_buffer, padding])
        mix_buffer[start_sample:end_sample] += audio

mix_start_time = None
SAMPLE_RATE = 44100

def getCurrentSample():
    if mix_start_time is None:
        return 0
    return int((time.time() - mix_start_time) * SAMPLE_RATE)

def saveRecording(filename="output.wav", sample_rate=44100):
    global mix_buffer
    with mix_lock:
        if len(mix_buffer) == 0:
            print("Nothing to save")
            return
        # normalize
        max_val = np.max(np.abs(mix_buffer))
        if max_val > 0:
            mix_buffer /= max_val
        sf.write(filename, mix_buffer, sample_rate)
        print(f"Saved to {filename}")
        mix_buffer = np.zeros((0, 2), dtype=np.float32)

def buildBoard(type="bg"):
    if type == "bg":
        return Pedalboard([
            LowpassFilter(cutoff_frequency_hz=1500),
            HighpassFilter(cutoff_frequency_hz=500),
            Resample(5000),
            Reverb(room_size=0.4, damping=0.8, wet_level=0.3),
            Delay(delay_seconds=0.3, feedback=0.3, mix=0.3),
        ])
    elif type == "lead":
        return Pedalboard([
            HighpassFilter(cutoff_frequency_hz=200),
            LowpassFilter(cutoff_frequency_hz=2000),
            Reverb(room_size=0.2, damping=0.9, wet_level=0.3),
            Delay(delay_seconds=0.3, feedback=0.3, mix=0.3),
        ])
    return Pedalboard()

board = buildBoard("bg")
lead_board = buildBoard("lead")

pending_audio = []
pending_lock = threading.Lock()
output_stream = None

def audio_callback(outdata, frames, time_info, status):
    with pending_lock:
        mixed = np.zeros((frames, 2), dtype=np.float32)
        still_pending = []
        for audio, pos in pending_audio:
            end = pos + frames
            if end <= len(audio):
                mixed += audio[pos:end]
                still_pending.append((audio, end))
            elif pos < len(audio):
                mixed[:len(audio)-pos] += audio[pos:]
        pending_audio[:] = still_pending

    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed /= max_val
    outdata[:] = mixed

def startOutputStream(sample_rate=44100):
    global output_stream
    output_stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=2,
        dtype='float32',
        blocksize=1024,
        callback=audio_callback
    )
    output_stream.start()

def queueAudio(processed, master_vol=0.25):
    with pending_lock:
        pending_audio.append((processed * master_vol, 0))

# ----- NOTES AND SHI -------

def playNote(frequency, duration=0.5, pan=0.0, volume=1.0, sample_rate=44100, is_lead=False):
    try:
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t) * volume

        fade_samples = int(sample_rate * 0.02)
        envelope = np.ones(len(wave))
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave *= envelope

        left = wave * (1 - max(0, pan))
        right = wave * (1 + min(0, pan))

        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left /= max_val
            right /= max_val

        stereo = np.ascontiguousarray(np.column_stack((left, right)).astype(np.float32))
        processed = board(stereo.T, sample_rate)
        processed = np.ascontiguousarray(processed.T)
        addToMix(processed, getCurrentSample())
        queueAudio(processed)
    except Exception as e:
        print(f"Audio error: {e}")

def playNoteAsync(frequency, duration, pan, volume=1.0, is_lead=False):
    global active_threads
    if active_threads >= MAX_THREADS:
        return
    active_threads += 1
    def play_and_release():
        global active_threads
        playNote(frequency, duration, pan, volume, 44100, is_lead)
        active_threads -= 1
    threading.Thread(target=play_and_release, daemon=True).start()

def playHeldNote(frequency, duration=3.0, pan=0.0, volume=1.0, sample_rate=44100):
    try:
        num_samples = int(sample_rate * duration)
        cycle_samples = int(sample_rate / frequency)
        t = np.arange(num_samples) / sample_rate
        wave = np.sin(2 * np.pi * frequency * t)
        
        fade_samples = int(sample_rate * 0.05)
        envelope = np.ones(num_samples)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        wave = wave * envelope * volume
        left = wave * (1 - max(0, pan))
        right = wave * (1 + min(0, pan))

        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left /= max_val
            right /= max_val

        stereo = np.ascontiguousarray(np.column_stack((left, right)).astype(np.float32))
        processed = board(stereo.T, sample_rate)
        processed = np.ascontiguousarray(processed.T)
        addToMix(processed, getCurrentSample())
        queueAudio(processed)
    except Exception as e:
        print(f"Audio error: {e}")

def playHeldNoteAsync(frequency, duration=3.0, pan=0.0, volume=1.0):
    threading.Thread(target=playHeldNote, args=(frequency, duration, pan, volume), daemon=True).start()

def playArpeggio(chord_notes, pan, volume, sample_rate=44100):
    try:
        note_duration = 0.15
        gap = note_duration * 2
        total_samples = int(sample_rate * (note_duration + gap) * len(chord_notes))
        left = np.zeros(total_samples)
        right = np.zeros(total_samples)

        for i, freq in enumerate(chord_notes):
            start = int(i * gap * sample_rate)
            t = np.linspace(0, note_duration, int(sample_rate * note_duration))
            wave = np.sin(2 * np.pi * freq * t) * volume
            end = start + len(wave)
            left[start:end] += wave * (1 - max(0, pan))
            right[start:end] += wave * (1 + min(0, pan))

        stereo = np.ascontiguousarray(np.column_stack((left, right)).astype(np.float32))
        processed = board(stereo.T, sample_rate)
        processed = np.ascontiguousarray(processed.T)
        addToMix(processed, getCurrentSample())

        queueAudio(processed)
    except Exception as e:
        print(f"Audio error: {e}")

def playArpeggioAsync(chord_notes, pan, volume):
    threading.Thread(target=playArpeggio, args=(chord_notes, pan, volume), daemon=True).start()

def playLeadPhrase(notes, final_note, note_duration=0.08, pan=0.0, volume=0.6, sample_rate=44100):
    try:
        final_duration = 1.0 if note_duration < 0.15 else 1.8
        
        note_samples = int(sample_rate * note_duration)
        final_samples = int(sample_rate * final_duration)
        total_samples = note_samples * len(notes) + final_samples
        
        left = np.zeros(total_samples)
        right = np.zeros(total_samples)

        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left /= max_val
            right /= max_val
        
        def add_note(freq, start_sample, n_samples, vol):
            t = np.arange(n_samples) / sample_rate
            wave = np.sin(2 * np.pi * freq * t) * vol
            wave += np.sin(2 * np.pi * freq * 2 * t) * vol * 0.4    # octave
            wave += np.sin(2 * np.pi * freq * 3 * t) * vol * 0.2    # fifth above octave
            wave += np.sin(2 * np.pi * freq * 0.5 * t) * vol * 0.15 # sub octave

            # normalize
            wave = wave / np.max(np.abs(wave)) * vol
            fade = int(sample_rate * 0.01)
            wave[:fade] *= np.linspace(0, 1, fade)
            wave[-fade:] *= np.linspace(1, 0, fade)
            end = start_sample + n_samples
            left[start_sample:end] += wave * (1 - max(0, pan))
            right[start_sample:end] += wave * (1 + min(0, pan))

        for i, freq in enumerate(notes):
            add_note(freq, i * note_samples, note_samples, volume)
        
        add_note(final_note, len(notes) * note_samples, final_samples, volume * 1.2)

        stereo = np.ascontiguousarray(np.column_stack((left, right)).astype(np.float32))
        processed = lead_board(stereo.T, sample_rate)
        processed = np.ascontiguousarray(processed.T)
        addToMix(processed, getCurrentSample())

        queueAudio(processed)
    except Exception as e:
        print(f"Lead audio error: {e}")

def playLeadPhraseAsync(notes, final_note, note_duration, pan=0.0, volume=0.6):
    threading.Thread(target=playLeadPhrase, args=(notes, final_note, note_duration, pan, volume), daemon=True).start()

class LeadVoice:
    def __init__(self):
        self.scale = [
            130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
            523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,
        ]
        self.motifs = [
            [7, 9, 11, 12],
            [12, 11, 9, 7, 5],
            [7, 9, 7, 5, 7],
            [9, 11, 12, 11, 9, 7],
            [7, 7, 9, 11, 12, 12],
            [12, 9, 7, 5, 4],
            [5, 7, 9, 11, 12, 14],
            [9, 8, 9, 11, 9, 7],
        ]
        self.style = random.choice(["solo"])
        self.motif_idx = 0
        self.phrase_count = 0
        self.last_note_idx = 7
        self.tension = 0.0
        random.shuffle(self.motifs)
        print(f"Lead style: {self.style}")

    def _apply_variation(self, motif):
        variation = random.choices(
            ["exact", "transpose", "reverse", "truncate", "extend", "skip"],
            weights=[2, 2, 2, 1, 2, 1]
        )[0]
        if variation == "exact":
            return motif
        elif variation == "transpose":
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            return [max(0, min(len(self.scale)-1, i + shift)) for i in motif]
        elif variation == "reverse":
            return list(reversed(motif))
        elif variation == "truncate":
            cut = random.randint(2, len(motif))
            return motif[:cut]
        elif variation == "extend":
            return motif + motif[-2:]
        elif variation == "skip":
            return motif[::2] if len(motif) > 2 else motif
        return motif

    def nextPhrase(self):
        self.phrase_count += 1
        self.tension = min(1.0, self.tension + 0.15)

        if self.phrase_count % random.randint(2, 3) == 0:
            self.motif_idx = (self.motif_idx + 1) % len(self.motifs)

        current_motif = self.motifs[self.motif_idx]

        if self.style == "solo":
            # release phrase when tension is high
            if self.tension > 0.7 and random.random() < 0.4:
                self.tension = 0.1
                notes = []
                idx = self.last_note_idx
                for _ in range(5):
                    notes.append(self.scale[idx])
                    idx = max(0, idx - random.randint(1, 2))
                final_note = self.scale[7]  # land on root C
                self.last_note_idx = 7
                return notes, final_note, 0.15

            fast_weight = max(1, int(self.tension * 4))
            slow_weight = max(1, int((1 - self.tension) * 4))
            character = random.choices(
                ["fast_run", "slow_melodic", "short_lick", "chromatic_run", "extended"],
                weights=[fast_weight, slow_weight, 2, 1, 2]
            )[0]

            if character == "fast_run":
                start_idx = max(0, min(len(self.scale)-3, self.last_note_idx + random.randint(-2, 2)))
                run_length = random.randint(6, 12)
                notes = []
                idx = start_idx
                for _ in range(run_length):
                    notes.append(self.scale[idx])
                    step = random.choices([1, 2, 1, 1, 3], weights=[4, 2, 4, 4, 1])[0]
                    idx = min(len(self.scale) - 1, idx + step)
                    if idx >= len(self.scale) - 1:
                        break
                if random.random() < 0.3:
                    for _ in range(random.randint(2, 4)):
                        idx = max(0, idx - random.randint(1, 2))
                        notes.append(self.scale[idx])
                self.last_note_idx = idx
                final_note = self.scale[current_motif[-1]]
                return notes, final_note, 0.07

            elif character == "slow_melodic":
                indices = self._apply_variation(current_motif)
                octave_shift = random.choice([0, 0, 7])
                indices = [min(len(self.scale)-1, i + octave_shift) for i in indices]
                notes = [self.scale[i] for i in indices[:-1]]
                final_note = self.scale[indices[-1]]
                self.last_note_idx = indices[-1]
                return notes, final_note, 0.22

            elif character == "short_lick":
                start_idx = max(0, min(len(self.scale)-4, self.last_note_idx + random.randint(-1, 1)))
                lick_patterns = [
                    [0, 2, 1, 3],
                    [0, -1, 1, 2],
                    [2, 1, 0, -1],
                    [0, 3, 1, 2],
                    [0, 1, -1, 2, 0],
                    [3, 1, 2, 0],
                    [0, 0, 2, 1],
                    [-1, 1, 3, 2, 0],
                    [0, 2, 4, 2, 1],
                    [1, -1, 2, -1, 3],
                ]
                pattern = random.choice(lick_patterns)
                notes = []
                for step in pattern:
                    idx = max(0, min(len(self.scale)-1, start_idx + step))
                    notes.append(self.scale[idx])
                self.last_note_idx = max(0, start_idx - 1)
                final_note = self.scale[self.last_note_idx]
                return notes, final_note, 0.1

            elif character == "chromatic_run":
                color_notes = {
                    "dorian": [138.59, 207.65, 277.18, 415.30, 554.37],
                    "locrian": [138.59, 185.00, 277.18, 369.99, 554.37],
                }
                mode = random.choice(["dorian", "locrian"])
                start_idx = max(0, min(len(self.scale)-3, self.last_note_idx + random.randint(-1, 1)))
                notes = [self.scale[start_idx]]
                color = random.choice(color_notes[mode])
                notes.append(color)
                notes.append(color * 1.5 if random.random() < 0.5 else self.scale[min(start_idx + 2, len(self.scale)-1)])
                notes.append(self.scale[min(start_idx + 1, len(self.scale)-1)])
                self.last_note_idx = min(start_idx + 1, len(self.scale)-1)
                final_note = self.scale[current_motif[-1]]
                return notes, final_note, 0.09

            elif character == "extended":
                all_notes = []
                num_sections = random.randint(2, 3)
                idx = self.last_note_idx
                for section in range(num_sections):
                    section_type = random.choice(["run", "lick", "melodic"])
                    if section_type == "run":
                        length = random.randint(4, 7)
                        for _ in range(length):
                            all_notes.append(self.scale[idx])
                            idx = min(len(self.scale)-1, idx + random.choices([1, 2, 1], weights=[4, 2, 4])[0])
                            if idx >= len(self.scale) - 1:
                                break
                    elif section_type == "lick":
                        start_idx = idx
                        pattern = random.choice([[0, 2, 1, 3], [0, -1, 1, 2], [2, 1, 0, -1]])
                        for step in pattern:
                            lidx = max(0, min(len(self.scale)-1, start_idx + step))
                            all_notes.append(self.scale[lidx])
                        idx = lidx
                    elif section_type == "melodic":
                        indices = self._apply_variation(random.choice(self.motifs))
                        all_notes.extend([self.scale[min(i, len(self.scale)-1)] for i in indices])
                        idx = min(indices[-1], len(self.scale)-1)
                self.last_note_idx = idx
                final_note = self.scale[current_motif[-1]]
                return all_notes, final_note, 0.09

        else:
            indices = self._apply_variation(current_motif)
            notes = [self.scale[i] for i in indices[:-1]]
            final_note = self.scale[indices[-1]]
            self.last_note_idx = indices[-1]
            return notes, final_note, 0.18