import unittest

from voice_mvp.text_chunks import SentenceChunker


class SentenceChunkerTest(unittest.TestCase):
    def test_yields_complete_sentences(self):
        chunker = SentenceChunker(min_chars=8)
        self.assertEqual(chunker.push("Hello there. Next"), ["Hello there."])
        self.assertEqual(chunker.flush(), ["Next"])

    def test_waits_for_minimum_size(self):
        chunker = SentenceChunker(min_chars=20)
        self.assertEqual(chunker.push("Hi. Still going. Enough now."), ["Hi. Still going."])
        self.assertEqual(chunker.flush(), ["Enough now."])

    def test_handles_newlines(self):
        chunker = SentenceChunker(min_chars=5)
        self.assertEqual(chunker.push("One\nTwo? Three"), ["One Two?"])
        self.assertEqual(chunker.flush(), ["Three"])


if __name__ == "__main__":
    unittest.main()

