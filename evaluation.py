from typing import List
import jiwer
from dataclasses import dataclass

wer_transformation = jiwer.Compose([
        # jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"ﭏ": "אל"}),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])

def process_words(gt, pred):
    return jiwer.process_words(
        gt,
        pred,
        reference_transform=wer_transformation,
        hypothesis_transform=wer_transformation,
    )

@dataclass
class TripleAlignmentChunk:
    type: str
    ref_start_idx: int
    ref_end_idx: int
    hyp1_start_idx: int
    hyp1_end_idx: int
    hyp2_start_idx: int
    hyp2_end_idx: int
    ref : str = None
    hyp1 : str = None
    hyp2 : str = None


def TripleAlignmentSummary(chunks : List[TripleAlignmentChunk]):
    corrections = 0
    corruptions = 0
    for chunk in chunks:
        if chunk.type == "correction":
            corrections += (chunk.ref_end_idx - chunk.ref_start_idx)
        elif chunk.type == "corruption":
            corruptions += (chunk.ref_end_idx - chunk.ref_start_idx)
    return {
        "corrections": corrections,
        "corruptions": corruptions,
        "total": chunks[-1].ref_end_idx-chunks[0].ref_start_idx,
    }


def intersect_alignments(out1, out2):
    triple_alignments = []
    i, j = 0, 0

    align1 = out1.alignments[0]
    align2 = out2.alignments[0]

    while i < len(align1) and j < len(align2):
        chunk1 = align1[i]
        chunk2 = align2[j]

        if chunk1.ref_end_idx <= chunk2.ref_start_idx or chunk2.ref_end_idx <= chunk1.ref_start_idx:
            # No overlap in ground truth ranges
            # Move to the next chunk in the sequence with the smaller end index
            if chunk1.ref_end_idx < chunk2.ref_end_idx:
                i += 1
            else:
                j += 1
        else:
            # Ranges overlap in ground truth
            intersect_start = max(chunk1.ref_start_idx, chunk2.ref_start_idx)
            intersect_end = min(chunk1.ref_end_idx, chunk2.ref_end_idx)

            if chunk1.type == "equal" and chunk2.type == "equal":
                new_type = "no_change"
            elif chunk1.type == "substitute" and chunk2.type == "equal":
                new_type = "correction"
            elif chunk1.type == "equal" and chunk2.type == "substitute":
                new_type = "corruption"
            elif chunk1.type == "substitute" and chunk2.type == "substitute":
                new_type = "no_change"  # TODO: check if align1 text == align2 text, if so, change to "no change", else "error replaced by a different error"
            else:
                new_type = "complex"

                hyp1_start_idx = max(chunk1.hyp_start_idx,
                                     intersect_start - chunk1.ref_start_idx + chunk1.hyp_start_idx)
                hyp1_end_idx = min(chunk1.hyp_end_idx, intersect_end - chunk1.ref_start_idx + chunk1.hyp_start_idx)
                hyp2_start_idx = max(chunk2.hyp_start_idx,
                                     intersect_start - chunk2.ref_start_idx + chunk2.hyp_start_idx)
                hyp2_end_idx = min(chunk2.hyp_end_idx, intersect_end - chunk2.ref_start_idx + chunk2.hyp_start_idx)

                triple_alignment = TripleAlignmentChunk(
                    type=new_type,
                    ref_start_idx=intersect_start,
                    ref_end_idx=intersect_end,
                    ref=out1.references[intersect_start:intersect_end],
                    hyp1_start_idx=hyp1_start_idx,
                    hyp1_end_idx=hyp1_end_idx,
                    hyp1=out1.hypotheses[hyp1_start_idx:hyp1_end_idx],
                    hyp2_start_idx=hyp2_start_idx,
                    hyp2_end_idx=hyp2_end_idx,
                    hyp2=out2.hypotheses[hyp2_start_idx:hyp2_end_idx],
                )

                triple_alignments.append(triple_alignment)

                # Move to the next chunk in the sequence with the smaller end index
                if chunk1.ref_end_idx == chunk2.ref_end_idx:
                    i += 1
                    j += 1
                elif chunk1.ref_end_idx < chunk2.ref_end_idx:
                    i += 1
                else:
                    j += 1

    return triple_alignments

def compare_triplet(gt, ocr, corrected):
    out1 = process_words(gt, ocr)
    out2 = process_words(gt, corrected)
    print(jiwer.visualize_alignment(out1, show_measures=False, skip_correct=False))
    print(jiwer.visualize_alignment(out2, show_measures=False, skip_correct=False))
    triple_alignments = intersect_alignments(out1, out2)
    print(triple_alignments)
    res = TripleAlignmentSummary(triple_alignments)
    print(res)
    return res

