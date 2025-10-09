"""
Display visual examples from the CelebV-HQ dataset with detailed information.
"""

from datasets import load_dataset
import json
from pathlib import Path

def show_examples():
    print("Loading CelebV-HQ dataset...")
    dataset = load_dataset("SwayStar123/CelebV-HQ")
    
    data = dataset['train']
    first_sample = data[0]
    
    # Get mappings
    appearance_labels = first_sample['meta_info']['appearance_mapping']
    action_labels = first_sample['meta_info']['action_mapping']
    
    clips = first_sample['clips']
    
    print("\n" + "=" * 80)
    print("📊 CELEBV-HQ DATASET OVERVIEW")
    print("=" * 80)
    print(f"\n Total Clips: {len(clips):,}")
    print(f" Appearance Attributes: {len(appearance_labels)}")
    print(f" Action Labels: {len(action_labels)}")
    
    print("\n🎭 Appearance Attributes:")
    print(f"    {', '.join(appearance_labels[:15])}...")
    
    print("\n🎬 Action Labels:")
    print(f"    {', '.join(action_labels[:15])}...")
    
    # Show 5 detailed examples
    print("\n" + "=" * 80)
    print("📹 DETAILED VIDEO CLIP EXAMPLES")
    print("=" * 80)
    
    examples = []
    clip_ids = list(clips.keys())[:5]
    
    for i, clip_id in enumerate(clip_ids, 1):
        clip_data = clips[clip_id]
        
        print(f"\n{'─' * 80}")
        print(f"Example {i}/5: {clip_id}")
        print(f"{'─' * 80}")
        
        print(f"\n📺 YouTube ID: {clip_data['ytb_id']}")
        print(f"⏱️  Duration: {clip_data['duration']['start_sec']:.2f}s to {clip_data['duration']['end_sec']:.2f}s")
        print(f"   Length: {clip_data['duration']['end_sec'] - clip_data['duration']['start_sec']:.2f} seconds")
        
        print(f"\n📐 Face Bounding Box:")
        bbox = clip_data['bbox']
        print(f"   Top: {bbox['top']:.3f}, Bottom: {bbox['bottom']:.3f}")
        print(f"   Left: {bbox['left']:.3f}, Right: {bbox['right']:.3f}")
        
        print(f"\n🎭 Appearance Attributes:")
        appearance_indices = clip_data['attributes']['appearance']
        if appearance_indices:
            selected_appearance = [appearance_labels[idx] for idx in appearance_indices[:10]]
            print(f"   {', '.join(selected_appearance)}")
            if len(appearance_indices) > 10:
                print(f"   ... and {len(appearance_indices) - 10} more")
        else:
            print("   None")
        
        print(f"\n🎬 Actions:")
        action_indices = clip_data['attributes']['action']
        if action_indices:
            selected_actions = [action_labels[idx] for idx in action_indices]
            print(f"   {', '.join(selected_actions)}")
        else:
            print("   None")
        
        print(f"\n😊 Emotions:")
        emotion = clip_data['attributes']['emotion']
        if emotion['sep_flag']:
            print(f"   {emotion['labels']}")
        else:
            print("   Not separated/available")
        
        # Store for JSON
        examples.append({
            'clip_id': clip_id,
            'youtube_id': clip_data['ytb_id'],
            'youtube_url': f"https://www.youtube.com/watch?v={clip_data['ytb_id']}",
            'start_time': clip_data['duration']['start_sec'],
            'end_time': clip_data['duration']['end_sec'],
            'duration_seconds': clip_data['duration']['end_sec'] - clip_data['duration']['start_sec'],
            'bbox': bbox,
            'appearance': [appearance_labels[idx] for idx in appearance_indices] if appearance_indices else [],
            'actions': [action_labels[idx] for idx in action_indices] if action_indices else [],
            'emotions': emotion['labels'] if emotion['sep_flag'] else None
        })
    
    # Save examples to JSON
    output_dir = Path("/root/blueberry-video/celebvhq_data")
    output_dir.mkdir(exist_ok=True)
    
    examples_file = output_dir / "video_examples.json"
    with open(examples_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"\n\n{'=' * 80}")
    print(f"💾 Examples saved to: {examples_file}")
    print("=" * 80)
    
    print("\n\n📝 DATASET SUMMARY:")
    print("─" * 80)
    print("CelebV-HQ is a high-quality celebrity video dataset with:")
    print("  • 35,666 video clips from YouTube")
    print("  • Face bounding boxes for tracking")
    print("  • 40 appearance attributes (hair, accessories, facial features)")
    print("  • 35 action labels (talking, smiling, eating, etc.)")
    print("  • Emotion annotations")
    print("  • Perfect for:")
    print("    - Video face generation")
    print("    - Facial attribute recognition")
    print("    - Action recognition")
    print("    - Emotion detection")
    print("    - Video-to-video translation")
    print("=" * 80)

if __name__ == "__main__":
    show_examples()

