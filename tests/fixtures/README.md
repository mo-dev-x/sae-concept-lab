# Test-owned bundle fixtures

These eight documents were the product's shipped FAKE placeholders until the
build switched to shipping real, measured concepts. They are kept HERE, under
`tests/`, because several backend-logic tests need *some* entry with known
directions, strengths and layers to exercise resolution and hook dispatch --
and that need is a property of the tests, not of the product.

Keeping them here means the product's shipped concept set can change without
breaking tests that were never really about the shipped set, and a reader can
no longer mistake them for something the product offers. Every one is
`provenance: "fake"`.
