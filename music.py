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

audio_lock = threading.Lock()
lead_lock = threading.Lock()

recording_buffer = []
recording_lock = threading.Lock()

def buildBoard(genre="lofi"):
    if genre == "lofi":
        return Pedalboard([
            LowpassFilter(cutoff_frequency_hz=1500),
            HighpassFilter(cutoff_frequency_hz=500),
            Resample(5000),
            Reverb(room_size=0.8, damping=0.7, wet_level=0.5),
            Delay(delay_seconds=0.3, feedback=0.3, mix=0.3),
        ])
    return Pedalboard([Reverb(room_size=0.5)])

board = buildBoard("lofi")

def addToRecording(processed_audio):
    with recording_lock:
        recording_buffer.append(processed_audio.copy())

def saveRecording(filename="output.wav", track_number=1, sample_rate=44100):
    with recording_lock:
        if not recording_buffer:
            print("Nothing to save")
            return
        full_audio = np.concatenate(recording_buffer, axis=0)
        sf.write(filename, full_audio, sample_rate)
        print(f"Saved to {filename}")
        recording_buffer.clear()
    
    try:
        audio = WAVE(filename)
        audio.add_tags()
        audio.tags.add(TIT2(encoding=3, text=f"Track {track_number}"))
        audio.tags.add(TPE1(encoding=3, text=f"Fart"))
        audio.tags.add(TALB(encoding=3, text=f"Fart Cat's Dungeon Synth"))
    except Exception as e:
        print(f"Metadata error: {e}")

def playNote(frequency, duration=0.5, pan=0.0, volume=1.0, sample_rate=44100, is_lead=False):
    lock = lead_lock if is_lead else audio_lock
    if not lock.acquire(blocking=False):
        return
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
        addToRecording(processed)
        with sd.OutputStream(samplerate=sample_rate, channels=2, dtype='float32') as stream:
            stream.write(processed)
    except Exception as e:
        print(f"Audio error: {e}")
    finally:
        lock.release()

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
    if not audio_lock.acquire(blocking=False):
        return
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
        addToRecording(processed)
        with sd.OutputStream(samplerate=sample_rate, channels=2, dtype='float32') as stream:
            stream.write(processed)
    except Exception as e:
        print(f"Audio error: {e}")
    finally:
        audio_lock.release()

def playHeldNoteAsync(frequency, duration=3.0, pan=0.0, volume=1.0):
    threading.Thread(target=playHeldNote, args=(frequency, duration, pan, volume), daemon=True).start()

def playArpeggio(chord_notes, pan, volume, sample_rate=44100):
    if not audio_lock.acquire(blocking=False):
        return
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
        addToRecording(processed)

        with sd.OutputStream(samplerate=sample_rate, channels=2, dtype='float32') as stream:
            stream.write(processed)
    except Exception as e:
        print(f"Audio error: {e}")
    finally:
        audio_lock.release()

def playArpeggioAsync(chord_notes, pan, volume):
    threading.Thread(target=playArpeggio, args=(chord_notes, pan, volume), daemon=True).start()

def playLeadPhrase(notes, final_note, note_duration=0.08, pan=0.0, volume=0.6, sample_rate=44100):
    if not lead_lock.acquire(blocking=False):
        return
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
        processed = board(stereo.T, sample_rate)
        processed = np.ascontiguousarray(processed.T)
        addToRecording(processed)

        with sd.OutputStream(samplerate=sample_rate, channels=2, dtype='float32') as stream:
            stream.write(processed)
    except Exception as e:
        print(f"Lead audio error: {e}")
    finally:
        lead_lock.release()

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
        self.style = random.choice(["lyrical", "solo"])

        self.base_motif = random.choice(self.motifs)
        print(f"Lead style: {self.style}, motif: {self.base_motif}")

    def _apply_variation(self, motif):
        variation = random.choice(["exact", "exact", "transpose", "reverse", "truncate"])
        if variation == "exact":
            return motif
        elif variation == "transpose":
            shift = random.choice([-2, -1, 1, 2])
            return [max(0, min(len(self.scale)-1, i + shift)) for i in motif]
        elif variation == "reverse":
            return list(reversed(motif))
        elif variation == "truncate":
            cut = random.randint(2, len(motif))
            return motif[:cut]
        return motif

    def nextPhrase(self):
        if self.style == "solo":
            start_idx = random.randint(0, 5)
            direction = 1
            run_length = random.randint(5, 10)
            
            notes = []
            idx = start_idx
            for _ in range(run_length):
                notes.append(self.scale[idx])
                step = random.choices([1, 2, 1, 1, 3], weights=[4, 2, 4, 4, 1])[0]
                idx = min(len(self.scale) - 1, idx + step)
                if idx >= len(self.scale) - 1:
                    break
            
            if random.random() < 0.3 and len(notes) > 3:
                drop_length = random.randint(2, 3)
                for _ in range(drop_length):
                    idx = max(0, idx - 1)
                    notes.append(self.scale[idx])
            
            final_note = self.scale[min(idx + 1, len(self.scale) - 1)]
            return notes, final_note, 0.08
        else:
            indices = self._apply_variation(self.base_motif)
            notes = [self.scale[i] for i in indices[:-1]]
            final_note = self.scale[indices[-1]]
            return notes, final_note, 0.18