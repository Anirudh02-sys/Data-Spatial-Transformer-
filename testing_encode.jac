node TokenNode {
    has word: str;
    has decoded: strc= "";
    has embedding: list[float] = [];
}



walker encoder {

    can traverse with entry {
        visit [-->](`?TokenNode);
    }

    can embed_token(word: str) -> list[float] {
        return [ord(char) % 256 / 256.0 for char in word];
    }


    can encode with entry {
        here.embedding = self.embed_token(here.word);
        print(here);
    }
}

walker decoder {

    can traverse with entry {
        visit [-->](`?TokenNode);
    }

    # Decoding function
     can decode_embedding(embedding: list[float]) -> str {
        return ''.join([chr(int(val * 256)) for val in embedding]);
    }

    can decode with  entry{
        here.decoded = self.decode_embedding(here.embedding);
        print(here);
    }
}

with entry {
    # Create TokenNode instances
    node1 = TokenNode(word="hello");
    node2 = TokenNode(word="world");

    # Link nodes
    node1 ++> node2;

    # Spawn encoder and decoder from the first node
    node1 spawn encoder();
    node1 spawn decoder();
}
