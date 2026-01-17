"""
Batch Generation Test for Fish-Speech Quantization Comparison

This script generates speech for 5 long sentences in 5 languages
using all available quantization levels (BF16, INT8, INT4) and saves
the results for comparison.

Run with: python tools/testing/batch_quant_test.py

Output structure:
    test_outputs/quant_comparison/
    ├── bf16/
    │   ├── english_1.wav
    │   ├── chinese_1.wav
    │   └── ...
    ├── int8/
    │   └── ...
    ├── int4/
    │   └── ...
    └── results.json
"""

import gc
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from loguru import logger

# Test sentences: 5 languages, each with a long sentence (~50-80 words)
TEST_SENTENCES = {
    "english": [
        "The advancement of artificial intelligence has transformed the way we interact with technology in our daily lives. From virtual assistants that help us manage our schedules to sophisticated algorithms that power recommendation systems, AI has become an integral part of modern society that continues to evolve at an unprecedented pace.",
        "Climate change represents one of the most pressing challenges facing humanity today. Rising sea levels, extreme weather events, and shifting ecosystems threaten communities around the world, requiring immediate action from governments, businesses, and individuals alike to reduce carbon emissions and protect our planet for future generations.",
        "The human brain remains one of the most complex and fascinating structures in the known universe. With approximately eighty-six billion neurons forming intricate networks of connections, it enables us to think, feel, remember, and create in ways that scientists are still working to fully understand and replicate.",
        "Literature has served as a mirror reflecting the human condition throughout the ages. From ancient epics to modern novels, stories have allowed us to explore different perspectives, cultures, and time periods, fostering empathy and understanding while preserving the wisdom and experiences of countless generations.",
        "Space exploration continues to push the boundaries of human knowledge and capability. As we venture further into the cosmos, discovering new planets, studying distant galaxies, and searching for signs of extraterrestrial life, we gain invaluable insights about our own origins and place in this vast universe.",
    ],
    "chinese": [
        "人工智能的快速发展正在深刻改变我们的生活方式和工作模式。从智能家居到自动驾驶汽车，从医疗诊断到金融分析，人工智能技术正在各个领域展现出强大的应用潜力，为人类社会带来前所未有的便利与效率，同时也引发了关于伦理和隐私的广泛讨论。",
        "中华文明源远流长，拥有五千多年的悠久历史。从黄河流域的远古文明到今天的现代化国家，中国人民创造了灿烂辉煌的文化遗产，包括哲学思想、文学艺术、科学技术等诸多领域的杰出成就，对世界文明的发展产生了深远的影响。",
        "教育是国家发展的基石，是民族振兴的希望。优质的教育不仅能够培养具有创新精神和实践能力的人才，还能促进社会公平、推动经济增长。在这个知识经济时代，终身学习已经成为每个人适应社会变革、实现自我价值的必要途径。",
        "环境保护是当今世界面临的重大挑战之一。随着工业化和城市化进程的加快，空气污染、水资源短缺、生物多样性丧失等问题日益严重。我们必须采取有效措施，践行绿色发展理念，建设生态文明，为子孙后代留下蓝天白云和绿水青山。",
        "科技创新是推动社会进步的核心动力。从蒸汽机的发明到互联网的普及，每一次重大技术突破都深刻改变了人类的生产生活方式。面对新一轮科技革命和产业变革，我们要加强基础研究，攻克关键核心技术，抢占科技制高点。",
    ],
    "japanese": [
        "日本の伝統文化は、長い歴史の中で独自の美意識と精神性を育んできました。茶道、華道、書道などの芸道は、単なる技術の習得ではなく、心を磨き、自然との調和を追求する修行の道として、現代においても多くの人々に受け継がれています。その奥深い世界観は、世界中の人々を魅了し続けています。",
        "人工知能技術の急速な発展は、私たちの社会に大きな変革をもたらしています。自動運転車、医療診断支援システム、音声認識技術など、様々な分野でAIの活用が進んでいます。一方で、プライバシーの保護や雇用への影響など、解決すべき課題も山積しており、技術と社会の調和が求められています。",
        "地球温暖化は、私たちの生活に深刻な影響を及ぼす環境問題です。異常気象の増加、海面上昇、生態系の変化など、その影響は既に世界各地で観測されています。持続可能な社会を実現するためには、再生可能エネルギーの普及や省エネルギー技術の開発が不可欠であり、一人一人の意識改革も重要です。",
        "読書は、知識を広げ、想像力を豊かにする素晴らしい習慣です。本を通じて、私たちは異なる時代や文化を体験し、様々な人物の人生を追体験することができます。デジタル時代においても、紙の本の魅力は色褪せることなく、多くの人々に知的な喜びと心の安らぎを与え続けています。",
        "宇宙探査は、人類の未知への探求心を象徴する壮大な挑戦です。月面着陸から半世紀以上が経過し、現在は火星への有人飛行や小惑星からのサンプルリターンなど、より野心的なミッションが計画されています。宇宙開発は、科学技術の発展を促進するだけでなく、地球外生命の探索という根源的な疑問にも答えを求めています。",
    ],
    "korean": [
        "인공지능 기술의 발전은 우리 사회의 모든 분야에 혁명적인 변화를 가져오고 있습니다. 의료, 교육, 금융, 제조업 등 다양한 산업에서 인공지능의 활용이 확대되면서, 효율성 향상과 새로운 가치 창출이 이루어지고 있습니다. 하지만 동시에 일자리 변화와 윤리적 문제에 대한 사회적 논의도 필요한 시점입니다.",
        "한국의 전통 문화는 수천 년의 역사 속에서 독특하고 아름다운 유산을 남겼습니다. 한글의 창제, 고려청자의 예술성, 조선시대의 유교 문화 등은 세계적으로 인정받는 문화유산입니다. 현대에 와서 케이팝과 한국 드라마가 전 세계적으로 큰 인기를 얻으며 한류 열풍을 일으키고 있습니다.",
        "기후 변화는 전 인류가 함께 해결해야 할 시급한 과제입니다. 지구 온난화로 인한 이상 기후 현상이 증가하고 있으며, 해수면 상승과 생태계 파괴가 심각한 수준에 이르고 있습니다. 지속 가능한 미래를 위해 신재생 에너지 개발과 탄소 배출 감축을 위한 국제적 협력이 그 어느 때보다 중요합니다.",
        "교육은 개인의 성장과 국가 발전의 근간이 됩니다. 창의적 사고력과 문제 해결 능력을 기르는 교육을 통해 미래 사회를 이끌어갈 인재를 양성해야 합니다. 디지털 기술의 발전으로 온라인 교육의 기회가 확대되면서, 평생 학습의 시대가 열리고 있습니다.",
        "우주 탐사는 인류의 무한한 호기심과 도전 정신을 보여주는 위대한 모험입니다. 달 착륙 이후 화성 탐사, 소행성 연구, 외계 생명체 탐색 등 다양한 프로젝트가 진행되고 있습니다. 우주 개발을 통해 얻어지는 과학 기술의 발전은 우리의 일상생활에도 많은 혜택을 가져다 줍니다.",
    ],
    "german": [
        "Die künstliche Intelligenz revolutioniert unsere Welt in einem beispiellosen Tempo. Von selbstfahrenden Autos über medizinische Diagnosen bis hin zu personalisierten Empfehlungssystemen verändert diese Technologie grundlegend, wie wir leben, arbeiten und miteinander kommunizieren. Dabei müssen wir sicherstellen, dass diese Entwicklung ethisch verantwortungsvoll und zum Wohle aller Menschen gestaltet wird.",
        "Die deutsche Kultur hat über Jahrhunderte hinweg bedeutende Beiträge zur Weltliteratur, Philosophie und Musik geleistet. Von Goethe und Schiller über Kant und Hegel bis zu Bach und Beethoven haben deutsche Denker und Künstler das geistige Erbe der Menschheit entscheidend geprägt und bereichert, wobei ihre Werke bis heute Menschen auf der ganzen Welt inspirieren.",
        "Der Klimawandel stellt eine der größten Herausforderungen unserer Zeit dar. Steigende Temperaturen, schmelzende Gletscher und häufigere Extremwetterereignisse bedrohen Ökosysteme und menschliche Gemeinschaften weltweit. Um eine lebenswerte Zukunft zu sichern, müssen wir entschlossen handeln und den Übergang zu erneuerbaren Energien beschleunigen.",
        "Bildung ist der Schlüssel zu persönlichem Wachstum und gesellschaftlichem Fortschritt. In einer sich schnell verändernden Welt wird lebenslanges Lernen immer wichtiger, um mit den neuesten Entwicklungen Schritt zu halten. Dabei spielen sowohl traditionelle Bildungseinrichtungen als auch moderne digitale Lernplattformen eine entscheidende Rolle.",
        "Die Raumfahrt eröffnet der Menschheit neue Horizonte und erweitert unser Verständnis des Universums. Mit jeder Mission lernen wir mehr über unseren Platz im Kosmos und die Möglichkeiten, die jenseits unseres Heimatplaneten liegen. Die Erforschung des Weltraums inspiriert Generationen von Wissenschaftlern und Träumern gleichermaßen.",
    ],
}


@dataclass
class GenerationResult:
    """Result of a single generation."""
    language: str
    sentence_idx: int
    text: str
    quantization: str
    generation_time: float
    audio_duration: float
    sample_rate: int
    output_path: str
    success: bool
    error: Optional[str] = None


@dataclass
class TestSummary:
    """Summary of the test run."""
    total_generations: int
    successful: int
    failed: int
    total_time: float
    results_by_quant: Dict[str, List[Dict]]
    avg_rtf_by_quant: Dict[str, float]


# Available models
MODELS = {
    "bf16": "openaudio-s1-mini",
    "int8": "openaudio-s1-mini-int8-torchao-20260116_182651",
    "int4": "openaudio-s1-mini-int4-g128-torchao-20260116_182842",
}


def save_audio(audio_data: np.ndarray, sample_rate: int, output_path: Path):
    """Save audio data to WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(str(output_path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())


def load_models(checkpoint_path: str, device: str = "cuda"):
    """Load LLaMA model and DAC decoder."""
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.dac.inference import load_model as load_decoder_model

    precision = torch.bfloat16

    logger.info(f"Loading model from {checkpoint_path}...")

    # Launch LLaMA queue (waits for init internally)
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
        compile=False,
    )
    logger.info("LLaMA model loaded")

    # Load DAC decoder (always from original checkpoint)
    decoder = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
        device=device,
    )
    logger.info("DAC decoder loaded")

    return llama_queue, decoder, precision


def generate_speech(
    llama_queue,
    decoder,
    precision,
    text: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    device: str = "cuda",
) -> Tuple[Optional[Tuple[int, np.ndarray]], float, Optional[str]]:
    """Generate speech from text."""
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest

    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder,
        precision=precision,
        compile=False,
    )

    req = ServeTTSRequest(
        text=text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        streaming=False,
    )

    start_time = time.time()
    audio_result = None
    error = None

    try:
        for result in engine.inference(req):
            if result.code == "final":
                audio_result = result.audio
            elif result.code == "error":
                error = str(result.error)
    except Exception as e:
        error = str(e)
        logger.error(f"Generation error: {e}")

    generation_time = time.time() - start_time

    return audio_result, generation_time, error


def run_batch_test(
    output_dir: Path = Path("test_outputs/quant_comparison"),
    models_to_test: Optional[List[str]] = None,
    languages_to_test: Optional[List[str]] = None,
    max_sentences_per_lang: int = 5,
):
    """Run batch generation test across all quantization levels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if models_to_test is None:
        models_to_test = list(MODELS.keys())

    if languages_to_test is None:
        languages_to_test = list(TEST_SENTENCES.keys())

    all_results: List[GenerationResult] = []
    results_by_quant: Dict[str, List[GenerationResult]] = {m: [] for m in models_to_test}

    total_start = time.time()

    for quant_name in models_to_test:
        checkpoint_path = f"checkpoints/{MODELS[quant_name]}"

        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {quant_name.upper()} quantization")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"{'='*60}\n")

        # Load model
        try:
            llama_queue, decoder, precision = load_models(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            continue

        # Record VRAM
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM used: {vram_used:.2f} GB")

        quant_output_dir = output_dir / quant_name
        quant_output_dir.mkdir(parents=True, exist_ok=True)

        for lang in languages_to_test:
            sentences = TEST_SENTENCES.get(lang, [])[:max_sentences_per_lang]

            for idx, text in enumerate(sentences, 1):
                logger.info(f"\n[{quant_name}] {lang} sentence {idx}/{len(sentences)}")
                logger.info(f"Text: {text[:100]}...")

                # Generate
                audio_result, gen_time, error = generate_speech(
                    llama_queue=llama_queue,
                    decoder=decoder,
                    precision=precision,
                    text=text,
                )

                # Save result
                output_path = quant_output_dir / f"{lang}_{idx}.wav"
                result = GenerationResult(
                    language=lang,
                    sentence_idx=idx,
                    text=text,
                    quantization=quant_name,
                    generation_time=gen_time,
                    audio_duration=0,
                    sample_rate=0,
                    output_path=str(output_path),
                    success=audio_result is not None,
                    error=error,
                )

                if audio_result:
                    sample_rate, audio_data = audio_result
                    audio_duration = len(audio_data) / sample_rate
                    result.audio_duration = audio_duration
                    result.sample_rate = sample_rate

                    # Save audio
                    save_audio(audio_data, sample_rate, output_path)

                    rtf = gen_time / audio_duration if audio_duration > 0 else 0
                    logger.info(f"  Generated {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f}x)")
                else:
                    logger.error(f"  Generation failed: {error}")

                all_results.append(result)
                results_by_quant[quant_name].append(result)

        # Clean up model
        del llama_queue, decoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Calculate summary
    successful = sum(1 for r in all_results if r.success)
    failed = len(all_results) - successful

    avg_rtf_by_quant = {}
    for quant_name, results in results_by_quant.items():
        successful_results = [r for r in results if r.success and r.audio_duration > 0]
        if successful_results:
            total_audio = sum(r.audio_duration for r in successful_results)
            total_gen = sum(r.generation_time for r in successful_results)
            avg_rtf_by_quant[quant_name] = total_gen / total_audio if total_audio > 0 else 0
        else:
            avg_rtf_by_quant[quant_name] = 0

    summary = TestSummary(
        total_generations=len(all_results),
        successful=successful,
        failed=failed,
        total_time=total_time,
        results_by_quant={k: [asdict(r) for r in v] for k, v in results_by_quant.items()},
        avg_rtf_by_quant=avg_rtf_by_quant,
    )

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total generations: {summary.total_generations}")
    logger.info(f"Successful: {summary.successful}")
    logger.info(f"Failed: {summary.failed}")
    logger.info(f"Total time: {summary.total_time:.2f}s")
    logger.info("")
    logger.info("Average Real-Time Factor by Quantization:")
    for quant_name, rtf in avg_rtf_by_quant.items():
        logger.info(f"  {quant_name}: {rtf:.3f}x")
    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch quantization test for Fish-Speech")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_outputs/quant_comparison"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=None,
        help="Models to test (default: all)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=list(TEST_SENTENCES.keys()),
        default=None,
        help="Languages to test (default: all)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=5,
        help="Maximum sentences per language",
    )

    args = parser.parse_args()

    run_batch_test(
        output_dir=args.output_dir,
        models_to_test=args.models,
        languages_to_test=args.languages,
        max_sentences_per_lang=args.max_sentences,
    )
