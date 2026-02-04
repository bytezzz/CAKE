use arrow_array::builder::{ArrayBuilder, GenericByteViewBuilder};
use arrow_array::types::{ByteArrayType, ByteViewType};
use arrow_array::{Array, BooleanArray, GenericByteArray, GenericByteViewArray, OffsetSizeTrait};
use arrow_buffer::ArrowNativeType;
use num::ToPrimitive;
pub fn filter_into_string_view<FROM, V>(
    byte_array: &GenericByteArray<FROM>,
    selection_arr: &BooleanArray,
) -> GenericByteViewArray<V>
where
    FROM: ByteArrayType,
    FROM::Offset: OffsetSizeTrait + ToPrimitive,
    V: ByteViewType<Native = FROM::Native>,
{
    let valid_count = selection_arr.values().count_set_bits();

    if valid_count == 0 {
        return GenericByteViewArray::new(vec![0].into(), vec![byte_array.values().clone()], None);
    }

    let offsets = byte_array.offsets();

    let can_reuse_buffer = match offsets.last() {
        Some(offset) => offset.as_usize() < u32::MAX as usize,
        None => true,
    };

    if can_reuse_buffer {
        let mut views_builder = GenericByteViewBuilder::<V>::with_capacity(valid_count);
        let str_values_buf = byte_array.values().clone();
        let block = views_builder.append_block(str_values_buf);

        for (i, w) in offsets.windows(2).enumerate() {
            if unsafe { !selection_arr.value_unchecked(i) } {
                continue;
            }
            let offset = w[0].as_usize();
            let end = w[1].as_usize();
            let length = end - offset;

            if byte_array.is_null(i) {
                views_builder.append_null();
            } else {
                // Safety: the input was a valid array so it valid UTF8 (if string). And
                // all offsets were valid
                unsafe { views_builder.append_view_unchecked(block, offset as u32, length as u32) }
            }
        }
        assert_eq!(views_builder.len(), valid_count);
        views_builder.finish()
    } else {
        panic!("too large arr");
    }
}

#[cfg(test)]
mod tests {
    use crate::cusfilter::filter_into_string_view;
    use arrow_array::{BooleanArray, StringArray, StringViewArray};

    #[test]
    fn test() {
        let arr = StringArray::from(vec!["Hello", "World", "From", "Penn"]);
        let select = BooleanArray::from(vec![true, false, false, true]);

        let result: StringViewArray = filter_into_string_view(&arr, &select);

        println!("{:?}", result);
    }
}
