import cv2
import numpy as np
import time
import random

import music

last_note_time = 0
song_start = time.time()
SONG_DURATION = 150

lead = music.LeadVoice()
lead_note_time = 0
lead_phrase_duration = 0

PHASES = [
    ("intro",  0,   20,  {"cooldown": 0.0, "volume_mult": 0.2}),
    ("verse",  20,  80,  {"cooldown": 0.0, "volume_mult": 0.4}),
    ("build",  80,  120, {"cooldown": 0.0, "volume_mult": 0.8}),
    ("outro",  120, 150, {"cooldown": 1.5, "volume_mult": 0.1}),
]
TRANSITION_DURATION = 8

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

cap = cv2.VideoCapture("http://192.168.1.59:5000/video")
cap.set(3, 320)
cap.set(4, 240)

def processImage(img):
    img = cv2.erode(img, (5,5), iterations = 1)
    img = cv2.dilate(img, (5,5), iterations = 2)

    return img

def getWhiteMask(img):
    lowerBound = np.array([0, 0, 180])
    upperBound = np.array([179, 40, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return processImage(cv2.inRange(hsv, lowerBound, upperBound))

def getOrangeMask(img):
    lowerBound = np.array([8, 100, 80])
    upperBound = np.array([20, 255, 200])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return processImage(cv2.inRange(hsv, lowerBound, upperBound))

def getBlobInfo(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])  # center x
    cy = int(M["m01"] / M["m00"])  # center y

    if area < 300:
        return None
    return cx, cy, area

def getPhaseAndParams(elapsed):
    for i, (phase, start, end, params) in enumerate(PHASES):
        if elapsed < end:
            time_left = end - elapsed
            if time_left < TRANSITION_DURATION and i + 1 < len(PHASES):
                next_phase, _, _, next_params = PHASES[i + 1]
                t = 1.0 - (time_left / TRANSITION_DURATION)
                blended = {
                    "cooldown": params["cooldown"] + t * (next_params["cooldown"] - params["cooldown"]),
                    "volume_mult": params["volume_mult"] + t * (next_params["volume_mult"] - params["volume_mult"]),
                }
                phase = next_phase if t > 0.5 else phase
                return phase, blended
            return phase, params
    return PHASES[-1][0], PHASES[-1][3]

chords_by_position = ["C", "Am", "F", "G"]

def main():
    global last_note_time, lead_note_time, lead_phrase_duration
    cooldown = 0.5

    last_left_chord = None
    last_right_chord = None
    last_left_note = None
    last_right_note = None

    left_chord_until = 0
    right_chord_until = 0
    left_note_until = 0
    right_note_until = 0

    left_chord = "C"
    right_chord = "C"

    music.mix_start_time = time.time()
    music.startOutputStream()
    while True:
        ret, img = cap.read()
        if not ret or img is None:
            continue
        white_mask = getWhiteMask(img)
        orange_mask = getOrangeMask(img)

        colored_mask = np.zeros_like(img)
        colored_mask[white_mask > 0] = [255, 255, 255]
        colored_mask[orange_mask > 0] = [0, 165, 255] 

        white_blob = getBlobInfo(white_mask)
        orange_blob = getBlobInfo(orange_mask)
        if white_blob:
            cx, cy, area = white_blob
            left_chord = chords_by_position[min(int(cx / 320 * len(chords_by_position)), len(chords_by_position) - 1)]
        elif orange_blob:
            cx, cy, area = orange_blob
            right_chord = chords_by_position[min(int(cx / 320 * len(chords_by_position)), len(chords_by_position) - 1)]

        elapsed = time.time() - song_start
        now = time.time()
        phase, params = getPhaseAndParams(elapsed)

        cv2.putText(colored_mask, phase.upper(), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(colored_mask, music.LeadVoice.lead_style.upper(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(colored_mask, "L: " + left_chord.upper() + " R: " + right_chord.upper(), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow('Dis shit (real)', colored_mask)
        if cv2.waitKey(1) == ord('q'):
            break
            
        if elapsed >= SONG_DURATION:
            print("Song complete!")
            music.saveRecording(f"track_{int(time.time())}.wav")
            break

        if phase in ("verse", "build"):
            if now > lead_phrase_duration:
                notes, final_note, note_dur = lead.nextPhrase(left_chord)
                music.playLeadPhraseAsync(notes, final_note, note_dur, pan=0.0, volume=1.5)
                phrase_length = note_dur * len(notes) + (1.0 if note_dur < 0.15 else 1.8)

                roll = random.random()
                if roll < 0.7:
                    rest = 0
                else:
                    rest = random.uniform(1, 2.0)

                lead_phrase_duration = now + phrase_length + rest

        if now - last_note_time < cooldown:
            continue

        scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        duration = 0.5 if random.random() > 0.2 else 1.0
        played = False

        if phase == "build":
            if white_blob:
                cx, cy, area = white_blob
                chord = music.chords[left_chord]

                if left_chord != last_left_chord or now > left_chord_until:
                    music.playArpeggioAsync(
                        chord, pan=-0.8,
                        volume=params["volume_mult"]
                    )
                    
                    last_left_chord = left_chord
                    left_chord_until = now + 0.65

                played = True
            if orange_blob:
                cx, cy, area = orange_blob
                chord = music.chords[right_chord]

                if right_chord != last_right_chord or now > right_chord_until:
                    music.playArpeggioAsync(
                        chord, pan=0.8,
                        volume=params["volume_mult"]
                    )

                    last_right_chord = right_chord
                    right_chord_until = now + 0.65

                played = True
        elif phase == "intro" or phase == "outro":
            if white_blob:
                cx, cy, area = white_blob
                note = scale[min(int(cx / 320 * len(scale)), len(scale) - 1)]
                volume = min(area / 5000, 1.0) * params["volume_mult"]

                if note != last_left_note or now > left_note_until:
                    music.playHeldNoteAsync(
                        note, duration=3,
                        pan=-0.8, volume=volume
                    )

                    last_left_note = note
                    left_note_until = now + 2.8

                played = True
            if orange_blob:
                cx, cy, area = orange_blob
                note = scale[min(int(cx / 320 * len(scale)), len(scale) - 1)]
                volume = min(area / 5000, 1.0) * params["volume_mult"]

                if note != last_right_note or now > right_note_until:
                    music.playHeldNoteAsync(
                        note, duration=3,
                        pan=0.8, volume=volume
                    )

                    last_right_note = note
                    right_note_until = now + 2.8

                played = True
        else:
            if white_blob:
                cx, cy, area = white_blob
                note = scale[min(int(cx / 320 * len(scale)), len(scale) - 1)]
                volume = min(area / 5000, 1.0) * params["volume_mult"]
                music.playHeldNoteAsync(note, duration, pan=-0.8, volume=volume)
                played = True
            if orange_blob:
                cx, cy, area = orange_blob
                note = scale[min(int(cx / 320 * len(scale)), len(scale) - 1)]
                volume = min(area / 5000, 1.0) * params["volume_mult"]
                music.playNoteAsync(note, duration, pan=0.8, volume=volume)
                played = True

        if played:
            last_note_time = now
            cooldown = params["cooldown"]
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()