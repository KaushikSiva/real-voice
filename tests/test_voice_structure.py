import unittest

from voice_mvp.server import (
    AssistantConfig,
    VoiceAssistantService,
    _parse_structured_voice_response,
    _safe_audio_name,
)


class VoiceStructureTest(unittest.TestCase):
    def test_parses_structured_filler_and_speech(self):
        filler, speech = _parse_structured_voice_response('{"filler":"hmm","speech":"Let me check that."}')
        self.assertEqual(filler, "hmm")
        self.assertEqual(speech, "Let me check that.")

    def test_falls_back_to_plain_speech(self):
        filler, speech = _parse_structured_voice_response("Yeah, that should work.")
        self.assertEqual(filler, "none")
        self.assertEqual(speech, "Yeah, that should work.")

    def test_rejects_unknown_filler(self):
        filler, speech = _parse_structured_voice_response('{"filler":"banana","speech":"Done."}')
        self.assertEqual(filler, "none")
        self.assertEqual(speech, "Done.")

    def test_safe_audio_name(self):
        self.assertEqual(_safe_audio_name("Hmm 1!"), "hmm1")

    def test_category_matches_filler_variants(self):
        service = VoiceAssistantService(
            AssistantConfig(canned_fillers="sure_1=Sure.|sure_2=Yeah.|hmm_1=Hmm.")
        )
        self.assertEqual(
            service._matching_canned_names(["sure"], {"sure_1", "sure_2", "hmm_1"}),
            ["sure_1", "sure_2"],
        )


if __name__ == "__main__":
    unittest.main()
