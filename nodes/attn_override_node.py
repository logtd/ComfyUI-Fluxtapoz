
def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
    

class FluxAttnOverrideNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "double_blocks": ("STRING", { "multiline": True }),
                "single_blocks": ("STRING", { "multiline": True }),
            }
        }

    RETURN_TYPES = ("ATTN_OVERRIDE",)
    FUNCTION = "build"

    CATEGORY = "fluxtapoz/attn"

    def build(self, double_blocks, single_blocks):
        double_block_map = set()
        for block in double_blocks.split(','):
            block = block.trim()
            if is_integer(block):
                double_block_map.add(block)

        single_block_map = set()
        for block in single_blocks.split(','):
            block = block.trim()
            if is_integer(block):
                single_block_map.add(block)

        return ({ "double": double_block_map, "single": single_block_map },)
    
